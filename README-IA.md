# Moonlight IA

Versión modificada de [Moonlight para Android](https://github.com/moonlight-stream/moonlight-android) con **mejora de imagen en la GPU del móvil**, pensada para recuperar nitidez cuando el bitrate es bajo (streaming con datos móviles).

Se instala **junto a** la Moonlight oficial (identificador `com.limelight.ia`, nombre "Moonlight IA").

## Ajuste nuevo

*Ajustes → Mejora de imagen*:

| Opción | Qué hace | Coste |
|---|---|---|
| Desactivada | Igual que Moonlight oficial | — |
| Nitidez (Snapdragon GSR) | Reescalado con detección de bordes y realce de nitidez | Muy bajo |
| Restauración IA, ligera (Anime4K S) | Red neuronal que elimina bloques y emborronado de compresión, + GSR | Medio |
| Restauración IA, calidad (Anime4K M) | Red más grande; si el stream es mucho menor que la pantalla, también reescala x2 con IA | Alto |

- Se desactiva sola en streams HDR (10 bits).
- Si la GPU no soporta algo, baja a un modo más simple. Si un fotograma falla, sigue mostrando la imagen sin filtro.
- El log de Moonlight muestra líneas `PostProcess:` con el modo efectivo y el tiempo por fotograma.

## Cómo funciona

El decodificador escribe en una textura de la GPU en vez de en la pantalla (`PostProcessor`). Cada fotograma pasa por los shaders y se dibuja en la ventana del stream. Los shaders de Anime4K usan el formato de ganchos de mpv, que interpreta `HookShader`.

Archivos:
- `app/src/main/java/com/limelight/binding/video/postprocess/` — motor de postprocesado
- `app/src/main/assets/postprocess/` — shaders y sus licencias
- `MediaCodecDecoderRenderer.java`, `PreferenceConfiguration.java`, `preferences.xml`, `strings.xml`, `arrays.xml` — integración

## Compilar

GitHub Actions (`.github/workflows/build-ia.yml`) compila la APK en cada push a `master`. Descárgala desde la pestaña *Actions* → última ejecución → *Artifacts* → **Moonlight-IA**.

La APK se firma con `ci/moonlight-ia.keystore` (contraseña `moonlight-ia`, o la variable `IA_KEYSTORE_PASSWORD`), para que cada versión nueva se instale encima de la anterior sin perder los ajustes. Es una clave solo para uso personal.

## Licencias

- Moonlight: GPL-3.0
- Anime4K (bloc97): MIT — `assets/postprocess/LICENSE-Anime4K.txt`
- Snapdragon Game Super Resolution (Qualcomm): BSD-3-Clause — `assets/postprocess/LICENSE-SnapdragonGSR.txt`
