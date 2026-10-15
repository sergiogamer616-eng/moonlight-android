package com.limelight.binding.video.postprocess;

import android.content.Context;
import android.graphics.SurfaceTexture;
import android.opengl.EGL14;
import android.opengl.EGLConfig;
import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGLExt;
import android.opengl.EGLSurface;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLES30;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Process;
import android.os.SystemClock;
import android.view.Surface;

import com.limelight.LimeLog;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * GPU post-processing between the video decoder and the screen.
 *
 * The decoder renders into a SurfaceTexture owned by this class. Every decoded frame is
 * copied into a regular texture, run through an optional chain of mpv-style hook shaders
 * (Anime4K) and finally drawn to the stream's window surface, optionally through
 * Snapdragon Game Super Resolution (edge-aware upscaling/sharpening).
 *
 * Internal textures use the image's natural row order (v = 0 is the top row), like mpv,
 * so the Anime4K networks see the image the same way they were trained.
 *
 * If anything fails during setup, {@link #create} returns null and the caller must use the
 * normal decoder output path. If a frame fails at runtime, the processor falls back to a
 * plain copy so the stream keeps working.
 */
public final class PostProcessor implements SurfaceTexture.OnFrameAvailableListener {

    public static final int MODE_OFF = 0;
    public static final int MODE_SHARPEN = 1;      // Snapdragon GSR only
    public static final int MODE_A4K_LIGHT = 2;    // Anime4K Restore CNN (S) + GSR
    public static final int MODE_A4K_QUALITY = 3;  // Anime4K Restore CNN (M) + Upscale (S, when needed) + GSR

    private static final String ASSET_DIR = "postprocess/";

    private final Context context;
    private final Surface window;
    private final int videoWidth, videoHeight;
    private final int requestedMode;

    private HandlerThread glThread;
    private Handler glHandler;

    private EGLDisplay eglDisplay = EGL14.EGL_NO_DISPLAY;
    private EGLContext eglContext = EGL14.EGL_NO_CONTEXT;
    private EGLSurface eglSurface = EGL14.EGL_NO_SURFACE;

    private int oesTexture;
    private SurfaceTexture surfaceTexture;
    private Surface inputSurface;
    private final float[] texMatrix = new float[16];

    private final AtomicInteger pendingFrames = new AtomicInteger();
    private volatile boolean released;

    private boolean es31;
    private boolean floatRenderable;
    private int vbo;

    // Programs
    private Program copyOesProgram;      // OES -> RGBA8 texture (flips to top-down rows)
    private Program presentProgram;      // texture -> window (bilinear, flips back)
    private Program presentOesProgram;   // OES -> window directly (fallback/simple path)
    private Program gsrProgram;          // texture -> window through Snapdragon GSR (may be null)

    private final List<HookPass> hookPasses = new ArrayList<>();
    private Tex mainTex;

    private int effectiveMode;
    private boolean runtimeFallback;

    // Stats
    private long statFrames, statNanos, statLastLog;

    private PostProcessor(Context context, Surface window, int videoWidth, int videoHeight, int mode) {
        this.context = context.getApplicationContext();
        this.window = window;
        this.videoWidth = videoWidth;
        this.videoHeight = videoHeight;
        this.requestedMode = mode;
    }

    /**
     * Creates a post-processor drawing into {@code window}. Returns null on any failure, in
     * which case the caller should give the decoder the window surface directly.
     */
    public static PostProcessor create(Context context, Surface window, int videoWidth, int videoHeight, int mode) {
        if (mode == MODE_OFF || window == null || !window.isValid()) {
            return null;
        }

        PostProcessor pp = new PostProcessor(context, window, videoWidth, videoHeight, mode);
        if (!pp.start()) {
            pp.release();
            return null;
        }
        return pp;
    }

    /** Surface the video decoder must render into. */
    public Surface getInputSurface() {
        return inputSurface;
    }

    public int getEffectiveMode() {
        return effectiveMode;
    }

    private boolean start() {
        glThread = new HandlerThread("PostProcess-GL", Process.THREAD_PRIORITY_DISPLAY);
        glThread.start();
        glHandler = new Handler(glThread.getLooper());

        final boolean[] ok = {false};
        final CountDownLatch latch = new CountDownLatch(1);
        glHandler.post(() -> {
            try {
                ok[0] = initGl();
            } catch (Throwable t) {
                LimeLog.warning("PostProcess: init failed: " + t);
                ok[0] = false;
            } finally {
                latch.countDown();
            }
        });

        try {
            if (!latch.await(5, TimeUnit.SECONDS)) {
                LimeLog.warning("PostProcess: init timed out");
                return false;
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
        return ok[0];
    }

    // ---------------------------------------------------------------------------------------
    // GL thread
    // ---------------------------------------------------------------------------------------

    private boolean initGl() {
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
        int[] version = new int[2];
        if (eglDisplay == EGL14.EGL_NO_DISPLAY || !EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) {
            LimeLog.warning("PostProcess: eglInitialize failed");
            return false;
        }

        int[] configAttribs = {
                EGL14.EGL_RED_SIZE, 8,
                EGL14.EGL_GREEN_SIZE, 8,
                EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_ALPHA_SIZE, 0,
                EGL14.EGL_RENDERABLE_TYPE, EGLExt.EGL_OPENGL_ES3_BIT_KHR,
                EGL14.EGL_SURFACE_TYPE, EGL14.EGL_WINDOW_BIT,
                EGL14.EGL_NONE
        };
        EGLConfig[] configs = new EGLConfig[1];
        int[] numConfigs = new int[1];
        if (!EGL14.eglChooseConfig(eglDisplay, configAttribs, 0, configs, 0, 1, numConfigs, 0) || numConfigs[0] == 0) {
            LimeLog.warning("PostProcess: no ES3 EGL config");
            return false;
        }

        int[] ctxAttribs = {EGL14.EGL_CONTEXT_CLIENT_VERSION, 3, EGL14.EGL_NONE};
        eglContext = EGL14.eglCreateContext(eglDisplay, configs[0], EGL14.EGL_NO_CONTEXT, ctxAttribs, 0);
        if (eglContext == EGL14.EGL_NO_CONTEXT) {
            LimeLog.warning("PostProcess: eglCreateContext failed");
            return false;
        }

        eglSurface = EGL14.eglCreateWindowSurface(eglDisplay, configs[0], window, new int[]{EGL14.EGL_NONE}, 0);
        if (eglSurface == EGL14.EGL_NO_SURFACE) {
            LimeLog.warning("PostProcess: eglCreateWindowSurface failed: 0x" + Integer.toHexString(EGL14.eglGetError()));
            return false;
        }

        if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
            LimeLog.warning("PostProcess: eglMakeCurrent failed");
            return false;
        }

        // Don't block on vsync: always show the newest processed frame (lowest latency)
        EGL14.eglSwapInterval(eglDisplay, 0);

        String glVersion = GLES20.glGetString(GLES20.GL_VERSION);
        String extensions = GLES20.glGetString(GLES20.GL_EXTENSIONS);
        if (glVersion == null || extensions == null) {
            return false;
        }
        es31 = glVersion.contains("OpenGL ES 3.1") || glVersion.contains("OpenGL ES 3.2");
        floatRenderable = glVersion.contains("OpenGL ES 3.2")
                || extensions.contains("GL_EXT_color_buffer_half_float")
                || extensions.contains("GL_EXT_color_buffer_float");
        LimeLog.info("PostProcess: " + glVersion + " (ES3.1=" + es31 + ", float RT=" + floatRenderable + ")");

        // Fullscreen triangle
        float[] tri = {-1f, -1f, 3f, -1f, -1f, 3f};
        FloatBuffer fb = ByteBuffer.allocateDirect(tri.length * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        fb.put(tri).position(0);
        int[] buf = new int[1];
        GLES20.glGenBuffers(1, buf, 0);
        vbo = buf[0];
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, vbo);
        GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, tri.length * 4, fb, GLES20.GL_STATIC_DRAW);

        // Decoder input
        int[] tex = new int[1];
        GLES20.glGenTextures(1, tex, 0);
        oesTexture = tex[0];
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTexture);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        surfaceTexture = new SurfaceTexture(oesTexture);
        surfaceTexture.setDefaultBufferSize(videoWidth, videoHeight);
        surfaceTexture.setOnFrameAvailableListener(this, glHandler);
        inputSurface = new Surface(surfaceTexture);

        // Basic programs (required)
        copyOesProgram = buildOesProgram(true);
        presentOesProgram = buildOesProgram(false);
        presentProgram = Program.build(vertexShader(300), PRESENT_FS, "present");
        if (copyOesProgram == null || presentOesProgram == null || presentProgram == null) {
            LimeLog.warning("PostProcess: basic shaders failed");
            return false;
        }

        // Pick the mode this device can actually run
        effectiveMode = requestedMode;
        if ((effectiveMode == MODE_A4K_LIGHT || effectiveMode == MODE_A4K_QUALITY) && !floatRenderable) {
            LimeLog.warning("PostProcess: no float render targets, using sharpen only");
            effectiveMode = MODE_SHARPEN;
        }

        if (es31) {
            gsrProgram = buildGsrProgram();
        }
        if (effectiveMode == MODE_SHARPEN && gsrProgram == null) {
            LimeLog.warning("PostProcess: GSR unavailable");
        }

        if (effectiveMode == MODE_A4K_LIGHT) {
            if (!loadHookChain(new String[]{"Anime4K_Restore_CNN_S.glsl"})) {
                effectiveMode = MODE_SHARPEN;
            }
        }
        else if (effectiveMode == MODE_A4K_QUALITY) {
            if (!loadHookChain(new String[]{"Anime4K_Restore_CNN_M.glsl", "Anime4K_Upscale_CNN_x2_S.glsl"})) {
                effectiveMode = MODE_SHARPEN;
            }
        }

        mainTex = new Tex(false);

        int err = GLES20.glGetError();
        if (err != GLES20.GL_NO_ERROR) {
            LimeLog.warning("PostProcess: GL error during init: 0x" + Integer.toHexString(err));
        }

        LimeLog.info("PostProcess: ready, mode " + requestedMode + " -> " + effectiveMode
                + ", video " + videoWidth + "x" + videoHeight + ", " + hookPasses.size() + " hook passes");
        return true;
    }

    @Override
    public void onFrameAvailable(SurfaceTexture st) {
        // Called on the GL thread (we passed glHandler)
        if (pendingFrames.getAndIncrement() == 0) {
            glHandler.post(this::renderFrame);
        }
    }

    private void renderFrame() {
        if (released) {
            return;
        }

        long start = SystemClock.elapsedRealtimeNanos();

        // Latch the newest frame and skip any older ones we couldn't keep up with
        int frames = pendingFrames.getAndSet(0);
        if (frames <= 0) {
            return;
        }
        try {
            for (int i = 0; i < frames; i++) {
                surfaceTexture.updateTexImage();
            }
        } catch (RuntimeException e) {
            LimeLog.warning("PostProcess: updateTexImage failed: " + e);
            return;
        }
        surfaceTexture.getTransformMatrix(texMatrix);

        int[] size = new int[1];
        EGL14.eglQuerySurface(eglDisplay, eglSurface, EGL14.EGL_WIDTH, size, 0);
        int outW = size[0];
        EGL14.eglQuerySurface(eglDisplay, eglSurface, EGL14.EGL_HEIGHT, size, 0);
        int outH = size[0];
        if (outW <= 0 || outH <= 0) {
            return;
        }

        try {
            if (runtimeFallback || effectiveMode == MODE_OFF) {
                presentOesDirect(outW, outH);
            }
            else {
                renderChain(outW, outH);
            }
        } catch (RuntimeException e) {
            LimeLog.warning("PostProcess: frame failed, falling back to plain output: " + e);
            runtimeFallback = true;
            presentOesDirect(outW, outH);
        }

        if (!EGL14.eglSwapBuffers(eglDisplay, eglSurface)) {
            int err = EGL14.eglGetError();
            if (err == EGL14.EGL_BAD_SURFACE || err == EGL14.EGL_BAD_NATIVE_WINDOW) {
                // Window is going away; the stream is stopping
                return;
            }
        }

        // Periodic stats in the log
        statFrames++;
        statNanos += SystemClock.elapsedRealtimeNanos() - start;
        long now = SystemClock.elapsedRealtime();
        if (now - statLastLog > 10000) {
            if (statFrames > 0) {
                LimeLog.info(String.format("PostProcess: %d frames, %.2f ms avg CPU submit, out %dx%d, mode %d%s",
                        statFrames, statNanos / 1e6 / statFrames, outW, outH, effectiveMode,
                        runtimeFallback ? " (fallback)" : ""));
            }
            statFrames = 0;
            statNanos = 0;
            statLastLog = now;
        }
    }

    private void presentOesDirect(int outW, int outH) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0);
        GLES20.glViewport(0, 0, outW, outH);
        presentOesProgram.use();
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTexture);
        GLES20.glUniform1i(presentOesProgram.loc("u_tex"), 0);
        GLES20.glUniformMatrix4fv(presentOesProgram.loc("u_texMatrix"), 1, false, texMatrix, 0);
        drawTriangle();
    }

    private void renderChain(int outW, int outH) {
        // 1. Decoder frame -> MAIN (RGBA8, top-down rows)
        mainTex.ensure(videoWidth, videoHeight);
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, mainTex.fbo);
        GLES20.glViewport(0, 0, videoWidth, videoHeight);
        copyOesProgram.use();
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTexture);
        GLES20.glUniform1i(copyOesProgram.loc("u_tex"), 0);
        GLES20.glUniformMatrix4fv(copyOesProgram.loc("u_texMatrix"), 1, false, texMatrix, 0);
        drawTriangle();

        // 2. Hook passes (Anime4K)
        final Map<String, Tex> named = new HashMap<>();
        named.put("MAIN", mainTex);
        named.put("NATIVE", mainTex);
        final int fOutW = outW, fOutH = outH;
        HookShader.SizeLookup sizes = name -> {
            if ("OUTPUT".equals(name)) {
                return new float[]{fOutW, fOutH};
            }
            Tex t = named.get("HOOKED".equals(name) ? "MAIN" : name);
            return t == null ? null : new float[]{t.width, t.height};
        };

        for (HookPass hp : hookPasses) {
            HookShader.Pass p = hp.pass;
            if (p.whenExpr != null && HookShader.evalRpn(p.whenExpr, sizes) == 0f) {
                continue;
            }
            Tex hooked = named.get("MAIN");
            int w = p.widthExpr != null ? Math.round(HookShader.evalRpn(p.widthExpr, sizes)) : hooked.width;
            int h = p.heightExpr != null ? Math.round(HookShader.evalRpn(p.heightExpr, sizes)) : hooked.height;
            if (w <= 0 || h <= 0) {
                throw new IllegalStateException("Bad pass size " + w + "x" + h + " in " + p.desc);
            }

            hp.output.ensure(w, h);
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, hp.output.fbo);
            GLES20.glViewport(0, 0, w, h);
            hp.program.use();
            for (int i = 0; i < hp.boundNames.size(); i++) {
                String name = hp.boundNames.get(i);
                Tex t = named.get("HOOKED".equals(name) ? "MAIN" : name);
                if (t == null) {
                    throw new IllegalStateException("Texture " + name + " not available for " + p.desc);
                }
                GLES20.glActiveTexture(GLES20.GL_TEXTURE0 + i);
                GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, t.id);
                GLES20.glUniform1i(hp.samplerLocs[i], i);
                GLES20.glUniform2f(hp.sizeLocs[i], t.width, t.height);
            }
            drawTriangle();

            named.put(p.target(), hp.output);
        }

        Tex result = named.get("MAIN");

        // 3. Present (GSR when available, otherwise bilinear)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0);
        GLES20.glViewport(0, 0, outW, outH);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, result.id);
        if (gsrProgram != null) {
            gsrProgram.use();
            GLES20.glUniform1i(gsrProgram.loc("ps0"), 0);
            GLES20.glUniform4f(gsrProgram.loc("ViewportInfo[0]"),
                    1f / result.width, 1f / result.height, result.width, result.height);
        }
        else {
            presentProgram.use();
            GLES20.glUniform1i(presentProgram.loc("u_tex"), 0);
        }
        drawTriangle();

        int err = GLES20.glGetError();
        if (err != GLES20.GL_NO_ERROR) {
            throw new IllegalStateException("GL error 0x" + Integer.toHexString(err));
        }
    }

    private void drawTriangle() {
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, vbo);
        GLES20.glEnableVertexAttribArray(0);
        GLES20.glVertexAttribPointer(0, 2, GLES20.GL_FLOAT, false, 0, 0);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 3);
    }

    // ---------------------------------------------------------------------------------------
    // Shader loading
    // ---------------------------------------------------------------------------------------

    private boolean loadHookChain(String[] files) {
        try {
            for (String file : files) {
                String src = readAsset(ASSET_DIR + file);
                for (HookShader.Pass p : HookShader.parse(src)) {
                    List<String> bound = new ArrayList<>(p.binds);
                    Map<String, String> uniforms = new HashMap<>();
                    String fs = HookShader.buildFragmentShader(p, bound, uniforms);
                    Program prog = Program.build(vertexShader(300), fs, p.desc);
                    if (prog == null) {
                        LimeLog.warning("PostProcess: failed to compile " + file + " / " + p.desc);
                        clearHookChain();
                        return false;
                    }
                    HookPass hp = new HookPass();
                    hp.pass = p;
                    hp.program = prog;
                    hp.boundNames = bound;
                    hp.samplerLocs = new int[bound.size()];
                    hp.sizeLocs = new int[bound.size()];
                    for (int i = 0; i < bound.size(); i++) {
                        hp.samplerLocs[i] = prog.loc("u_tex" + i);
                        hp.sizeLocs[i] = prog.loc("u_size" + i);
                    }
                    // Intermediate feature maps need signed values -> half float.
                    // The final image (written back to MAIN) fits in RGBA8.
                    hp.output = new Tex(!"MAIN".equals(p.target()));
                    hookPasses.add(hp);
                }
            }
            return true;
        } catch (IOException | RuntimeException e) {
            LimeLog.warning("PostProcess: failed to load shaders: " + e);
            clearHookChain();
            return false;
        }
    }

    private void clearHookChain() {
        for (HookPass hp : hookPasses) {
            hp.program.delete();
            hp.output.delete();
        }
        hookPasses.clear();
    }

    private Program buildGsrProgram() {
        try {
            String src = readAsset(ASSET_DIR + "sgsr1_mobile.frag");
            return Program.build(vertexShader(310), src, "SGSR");
        } catch (IOException e) {
            LimeLog.warning("PostProcess: GSR shader missing: " + e);
            return null;
        }
    }

    /** OES sampling program; copy=true writes top-down rows into an FBO, false draws to the window. */
    private Program buildOesProgram(boolean copy) {
        String flip = copy ? "vec2(v_uv.x, 1.0 - v_uv.y)" : "v_uv";
        String fs300 = "#version 300 es\n"
                + "#extension GL_OES_EGL_image_external_essl3 : require\n"
                + "precision mediump float;\n"
                + "uniform samplerExternalOES u_tex;\n"
                + "uniform highp mat4 u_texMatrix;\n"
                + "in highp vec2 v_uv;\n"
                + "out vec4 fragColor;\n"
                + "void main() {\n"
                + "    highp vec2 tc = (u_texMatrix * vec4(" + flip + ", 0.0, 1.0)).xy;\n"
                + "    fragColor = vec4(texture(u_tex, tc).rgb, 1.0);\n"
                + "}\n";
        Program p = Program.build(vertexShader(300), fs300, copy ? "copyOES" : "presentOES");
        if (p != null) {
            return p;
        }

        // Some drivers only expose external textures to ESSL 1.00
        String fs100 = "#extension GL_OES_EGL_image_external : require\n"
                + "precision mediump float;\n"
                + "uniform samplerExternalOES u_tex;\n"
                + "uniform highp mat4 u_texMatrix;\n"
                + "varying highp vec2 v_uv;\n"
                + "void main() {\n"
                + "    highp vec2 tc = (u_texMatrix * vec4(" + flip + ", 0.0, 1.0)).xy;\n"
                + "    gl_FragColor = vec4(texture2D(u_tex, tc).rgb, 1.0);\n"
                + "}\n";
        return Program.build(vertexShader(100), fs100, copy ? "copyOES100" : "presentOES100");
    }

    private static String vertexShader(int version) {
        if (version == 100) {
            return "attribute vec2 a_pos;\n"
                    + "varying highp vec2 v_uv;\n"
                    + "void main() {\n"
                    + "    v_uv = a_pos * 0.5 + 0.5;\n"
                    + "    gl_Position = vec4(a_pos, 0.0, 1.0);\n"
                    + "}\n";
        }
        return "#version " + version + " es\n"
                + "layout(location = 0) in vec2 a_pos;\n"
                + "out highp vec2 v_uv;\n"
                + "void main() {\n"
                + "    v_uv = a_pos * 0.5 + 0.5;\n"
                + "    gl_Position = vec4(a_pos, 0.0, 1.0);\n"
                + "}\n";
    }

    // Internal textures are top-down; flip back when drawing to the window
    private static final String PRESENT_FS = "#version 300 es\n"
            + "precision mediump float;\n"
            + "uniform sampler2D u_tex;\n"
            + "in highp vec2 v_uv;\n"
            + "out vec4 fragColor;\n"
            + "void main() {\n"
            + "    fragColor = vec4(texture(u_tex, vec2(v_uv.x, 1.0 - v_uv.y)).rgb, 1.0);\n"
            + "}\n";

    private String readAsset(String path) throws IOException {
        try (InputStream in = context.getAssets().open(path)) {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buf = new byte[16384];
            int n;
            while ((n = in.read(buf)) > 0) {
                out.write(buf, 0, n);
            }
            return new String(out.toByteArray(), StandardCharsets.UTF_8);
        }
    }

    // ---------------------------------------------------------------------------------------
    // Teardown
    // ---------------------------------------------------------------------------------------

    /** Must be called after the decoder that renders into {@link #getInputSurface()} is released. */
    public void release() {
        released = true;
        if (glHandler == null) {
            return;
        }

        final CountDownLatch latch = new CountDownLatch(1);
        glHandler.post(() -> {
            try {
                releaseGl();
            } catch (Throwable t) {
                LimeLog.warning("PostProcess: release error: " + t);
            } finally {
                latch.countDown();
            }
        });
        try {
            latch.await(2, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        glThread.quitSafely();
        glHandler = null;
    }

    private void releaseGl() {
        if (surfaceTexture != null) {
            surfaceTexture.setOnFrameAvailableListener(null);
        }
        if (eglDisplay != EGL14.EGL_NO_DISPLAY && eglContext != EGL14.EGL_NO_CONTEXT) {
            clearHookChain();
            if (mainTex != null) mainTex.delete();
            for (Program p : new Program[]{copyOesProgram, presentProgram, presentOesProgram, gsrProgram}) {
                if (p != null) p.delete();
            }
            if (oesTexture != 0) GLES20.glDeleteTextures(1, new int[]{oesTexture}, 0);
            if (vbo != 0) GLES20.glDeleteBuffers(1, new int[]{vbo}, 0);
        }
        if (inputSurface != null) {
            inputSurface.release();
            inputSurface = null;
        }
        if (surfaceTexture != null) {
            surfaceTexture.release();
            surfaceTexture = null;
        }
        if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
            EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT);
            if (eglSurface != EGL14.EGL_NO_SURFACE) EGL14.eglDestroySurface(eglDisplay, eglSurface);
            if (eglContext != EGL14.EGL_NO_CONTEXT) EGL14.eglDestroyContext(eglDisplay, eglContext);
            EGL14.eglReleaseThread();
        }
        eglSurface = EGL14.EGL_NO_SURFACE;
        eglContext = EGL14.EGL_NO_CONTEXT;
        eglDisplay = EGL14.EGL_NO_DISPLAY;
    }

    // ---------------------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------------------

    private static final class HookPass {
        HookShader.Pass pass;
        Program program;
        List<String> boundNames;
        int[] samplerLocs;
        int[] sizeLocs;
        Tex output;
    }

    /** Render target texture (RGBA8 or RGBA16F) with its framebuffer. */
    private static final class Tex {
        final boolean isFloat;
        int id, fbo, width, height;

        Tex(boolean isFloat) {
            this.isFloat = isFloat;
        }

        void ensure(int w, int h) {
            if (id != 0 && w == width && h == height) {
                return;
            }
            delete();
            int[] ids = new int[1];
            GLES20.glGenTextures(1, ids, 0);
            id = ids[0];
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, id);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
            if (isFloat) {
                GLES30.glTexStorage2D(GLES20.GL_TEXTURE_2D, 1, GLES30.GL_RGBA16F, w, h);
            }
            else {
                GLES30.glTexStorage2D(GLES20.GL_TEXTURE_2D, 1, GLES30.GL_RGBA8, w, h);
            }
            GLES20.glGenFramebuffers(1, ids, 0);
            fbo = ids[0];
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo);
            GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0,
                    GLES20.GL_TEXTURE_2D, id, 0);
            int status = GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER);
            if (status != GLES30.GL_FRAMEBUFFER_COMPLETE) {
                throw new IllegalStateException("Framebuffer incomplete (" + (isFloat ? "RGBA16F" : "RGBA8")
                        + " " + w + "x" + h + "): 0x" + Integer.toHexString(status));
            }
            width = w;
            height = h;
        }

        void delete() {
            if (fbo != 0) GLES20.glDeleteFramebuffers(1, new int[]{fbo}, 0);
            if (id != 0) GLES20.glDeleteTextures(1, new int[]{id}, 0);
            fbo = 0;
            id = 0;
            width = 0;
            height = 0;
        }
    }

    /** Linked shader program with cached uniform locations. */
    private static final class Program {
        final int id;
        private final Map<String, Integer> locs = new HashMap<>();

        private Program(int id) {
            this.id = id;
        }

        static Program build(String vs, String fs, String name) {
            int v = compile(GLES20.GL_VERTEX_SHADER, vs, name);
            if (v == 0) return null;
            int f = compile(GLES20.GL_FRAGMENT_SHADER, fs, name);
            if (f == 0) {
                GLES20.glDeleteShader(v);
                return null;
            }
            int p = GLES20.glCreateProgram();
            GLES20.glAttachShader(p, v);
            GLES20.glAttachShader(p, f);
            GLES20.glBindAttribLocation(p, 0, "a_pos");
            GLES20.glLinkProgram(p);
            GLES20.glDeleteShader(v);
            GLES20.glDeleteShader(f);
            int[] ok = new int[1];
            GLES20.glGetProgramiv(p, GLES20.GL_LINK_STATUS, ok, 0);
            if (ok[0] == 0) {
                LimeLog.warning("PostProcess: link failed (" + name + "): " + GLES20.glGetProgramInfoLog(p));
                GLES20.glDeleteProgram(p);
                return null;
            }
            return new Program(p);
        }

        private static int compile(int type, String src, String name) {
            int s = GLES20.glCreateShader(type);
            GLES20.glShaderSource(s, src);
            GLES20.glCompileShader(s);
            int[] ok = new int[1];
            GLES20.glGetShaderiv(s, GLES20.GL_COMPILE_STATUS, ok, 0);
            if (ok[0] == 0) {
                LimeLog.warning("PostProcess: compile failed (" + name + "): " + GLES20.glGetShaderInfoLog(s));
                GLES20.glDeleteShader(s);
                return 0;
            }
            return s;
        }

        void use() {
            GLES20.glUseProgram(id);
        }

        int loc(String name) {
            Integer l = locs.get(name);
            if (l == null) {
                l = GLES20.glGetUniformLocation(id, name);
                locs.put(name, l);
            }
            return l;
        }

        void delete() {
            GLES20.glDeleteProgram(id);
        }
    }
}
