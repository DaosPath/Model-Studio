# Model Studio

Aplicación de escritorio Tauri para ejecutar y gestionar modelos de lenguaje (GGUF) y difusión.

Actualmente integra un servidor de inferencia persistente, base de datos SQLite para el historial de conversaciones y galería de imágenes, y un servidor API local compatible con OpenAI en el puerto `1234`.

## Desarrollo

1. Instala las dependencias del frontend:
   ```powershell
   pnpm install
   ```

2. Coloca los ejecutables de inferencia (`llama-diffusion-cli.exe`, `sd-cli.exe` y sus DLLs correspondientes) dentro de la carpeta `runtime/` (estos binarios están ignorados en Git por su peso).

3. Inicia la aplicación en modo desarrollo:
   ```powershell
   pnpm tauri dev
   ```

Los modelos GGUF y Checkpoints se pueden seleccionar dinámicamente desde el panel de la aplicación.
