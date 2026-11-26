import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
    base: '/workbench/',
    plugins: [react()],
    server: {
        host: '127.0.0.1',
        port: 5174,
        strictPort: true,
        proxy: {
            '/api': { target: 'http://127.0.0.1:8787', changeOrigin: true },
            '/pipeline-viewer': { target: 'http://127.0.0.1:8787', changeOrigin: true },
        },
    },
});
