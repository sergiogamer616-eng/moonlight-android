package com.limelight.binding.video.postprocess;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Map;

/**
 * Parser for the subset of mpv's user shader ("//!HOOK") format used by Anime4K.
 *
 * Supported directives: DESC, HOOK (MAIN only), BIND, SAVE, WIDTH, HEIGHT, WHEN, COMPONENTS.
 * Passes that hook anything other than MAIN (e.g. PREKERNEL) are skipped.
 */
public final class HookShader {

    /** One render pass of a hook shader. */
    public static final class Pass {
        public String desc = "";
        public String hook = "MAIN";
        public final List<String> binds = new ArrayList<>();
        public String save;            // null = overwrite the hooked texture (MAIN)
        public String widthExpr;       // RPN, null = same size as the hooked texture
        public String heightExpr;
        public String whenExpr;        // RPN, null = always
        public int components = 4;
        public String body = "";

        /** Name of the texture this pass writes. */
        public String target() {
            return save != null ? save : hook;
        }
    }

    private HookShader() {}

    /** Parses a shader file into passes. Passes hooking textures other than MAIN are dropped. */
    public static List<Pass> parse(String source) {
        List<Pass> passes = new ArrayList<>();
        Pass current = null;
        StringBuilder body = new StringBuilder();
        boolean inHeader = false;

        for (String rawLine : source.split("\n", -1)) {
            String line = rawLine.endsWith("\r") ? rawLine.substring(0, rawLine.length() - 1) : rawLine;
            String trimmed = line.trim();

            if (trimmed.startsWith("//!")) {
                if (!inHeader) {
                    // A new pass header starts: close the previous pass
                    if (current != null) {
                        current.body = body.toString();
                        passes.add(current);
                    }
                    current = new Pass();
                    body.setLength(0);
                    inHeader = true;
                }

                String directive = trimmed.substring(3).trim();
                int space = directive.indexOf(' ');
                String key = space < 0 ? directive : directive.substring(0, space);
                String value = space < 0 ? "" : directive.substring(space + 1).trim();

                switch (key) {
                    case "DESC": current.desc = value; break;
                    case "HOOK": current.hook = value; break;
                    case "BIND": current.binds.add(value); break;
                    case "SAVE": current.save = value; break;
                    case "WIDTH": current.widthExpr = value; break;
                    case "HEIGHT": current.heightExpr = value; break;
                    case "WHEN": current.whenExpr = value; break;
                    case "COMPONENTS":
                        try {
                            current.components = Integer.parseInt(value);
                        } catch (NumberFormatException ignored) {
                        }
                        break;
                    default:
                        // OFFSET, COMPUTE, TEXTURE, etc. are not used by the shaders we ship
                        break;
                }
            }
            else {
                inHeader = false;
                if (current != null) {
                    body.append(line).append('\n');
                }
                // Text before the first header (license comments) is ignored
            }
        }

        if (current != null) {
            current.body = body.toString();
            passes.add(current);
        }

        List<Pass> result = new ArrayList<>();
        for (Pass p : passes) {
            if ("MAIN".equals(p.hook)) {
                result.add(p);
            }
        }
        return result;
    }

    /** Size lookup used when evaluating WIDTH/HEIGHT/WHEN expressions. */
    public interface SizeLookup {
        /** Returns {width, height} of a named texture, or null if unknown. */
        float[] size(String name);
    }

    /**
     * Evaluates an mpv RPN expression such as "MAIN.w 2 *" or
     * "OUTPUT.w MAIN.w / 1.200 > OUTPUT.h MAIN.h / 1.200 > *".
     */
    public static float evalRpn(String expr, SizeLookup sizes) {
        Deque<Float> stack = new ArrayDeque<>();
        for (String tok : expr.trim().split("\\s+")) {
            if (tok.isEmpty()) {
                continue;
            }
            switch (tok) {
                case "+": { float b = stack.pop(), a = stack.pop(); stack.push(a + b); break; }
                case "-": { float b = stack.pop(), a = stack.pop(); stack.push(a - b); break; }
                case "*": { float b = stack.pop(), a = stack.pop(); stack.push(a * b); break; }
                case "/": { float b = stack.pop(), a = stack.pop(); stack.push(a / b); break; }
                case ">": { float b = stack.pop(), a = stack.pop(); stack.push(a > b ? 1f : 0f); break; }
                case "<": { float b = stack.pop(), a = stack.pop(); stack.push(a < b ? 1f : 0f); break; }
                case "=": { float b = stack.pop(), a = stack.pop(); stack.push(a == b ? 1f : 0f); break; }
                case "!": { float a = stack.pop(); stack.push(a == 0f ? 1f : 0f); break; }
                default: {
                    int dot = tok.lastIndexOf('.');
                    if (dot > 0 && (tok.endsWith(".w") || tok.endsWith(".width")
                            || tok.endsWith(".h") || tok.endsWith(".height"))) {
                        String name = tok.substring(0, dot);
                        float[] s = sizes.size(name);
                        if (s == null) {
                            throw new IllegalArgumentException("Unknown texture in expression: " + tok);
                        }
                        stack.push(tok.charAt(dot + 1) == 'w' ? s[0] : s[1]);
                    }
                    else {
                        stack.push(Float.parseFloat(tok));
                    }
                    break;
                }
            }
        }
        if (stack.size() != 1) {
            throw new IllegalArgumentException("Bad RPN expression: " + expr);
        }
        return stack.pop();
    }

    /**
     * Builds a GLSL ES 3.00 fragment shader for one pass. Every bound texture gets the
     * mpv-style helper macros NAME_raw/NAME_tex/NAME_texOff/NAME_pos/NAME_size/NAME_pt.
     * "HOOKED" is an alias for the hooked texture (MAIN).
     *
     * @param uniformNames receives, in order, the sampler uniform name for each bound texture
     */
    public static String buildFragmentShader(Pass pass, List<String> boundTextures,
                                             Map<String, String> uniformNames) {
        StringBuilder sb = new StringBuilder();
        sb.append("#version 300 es\n");
        sb.append("precision mediump float;\n");
        sb.append("precision mediump sampler2D;\n");
        sb.append("in highp vec2 v_uv;\n");
        sb.append("out vec4 fragColor;\n");

        int idx = 0;
        for (String name : boundTextures) {
            String u = "u_tex" + idx;
            String s = "u_size" + idx;
            uniformNames.put(name, u);
            sb.append("uniform sampler2D ").append(u).append(";\n");
            sb.append("uniform highp vec2 ").append(s).append(";\n");
            sb.append("#define ").append(name).append("_raw ").append(u).append('\n');
            sb.append("#define ").append(name).append("_pos v_uv\n");
            sb.append("#define ").append(name).append("_size ").append(s).append('\n');
            sb.append("#define ").append(name).append("_pt (vec2(1.0) / ").append(s).append(")\n");
            sb.append("#define ").append(name).append("_tex(pos) texture(").append(u).append(", (pos))\n");
            sb.append("#define ").append(name).append("_texOff(off) texture(").append(u)
                    .append(", v_uv + (vec2(1.0) / ").append(s).append(") * (off))\n");
            idx++;
        }

        sb.append(pass.body);
        sb.append("\nvoid main() {\n    fragColor = hook();\n}\n");
        return sb.toString();
    }
}
