import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { viteStaticCopy } from 'vite-plugin-static-copy'

export default defineConfig({
  base: './',
  plugins: [
    react(),
    viteStaticCopy({
      targets: [
        { src: 'model/*', dest: 'model' },
        { src: 'wasm/*', dest: 'wasm' },
      ],
    }),
  ],
  build: {
    target: 'es2022',
    sourcemap: false,
  },
})
