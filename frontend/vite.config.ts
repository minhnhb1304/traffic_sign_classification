import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // base: '/<repo-name>/', // ONLY for GitHub Pages project sites
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
})
