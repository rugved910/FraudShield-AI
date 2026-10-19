/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Dark financial-workstation palette
        navy: {
          950: "#0a0f1e",
          900: "#0d1425",
          850: "#111a30",
          800: "#16203a",
          700: "#1e2c4d",
          600: "#2a3c66",
        },
        signal: {
          green: "#34d399",
          amber: "#fbbf24",
          red: "#f87171",
          blue: "#38bdf8",
        },
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};
