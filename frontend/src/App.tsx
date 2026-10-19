import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import Analyzer from "./pages/Analyzer";
import BatchAnalysis from "./pages/BatchAnalysis";
import Performance from "./pages/Performance";
import About from "./pages/About";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/analyzer" element={<Analyzer />} />
        <Route path="/batch" element={<BatchAnalysis />} />
        <Route path="/performance" element={<Performance />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </Layout>
  );
}
