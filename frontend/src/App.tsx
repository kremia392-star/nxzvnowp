import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { HomePage } from './pages/HomePage'
import { ArchitecturePage } from './pages/ArchitecturePage'
import { SparsityPage } from './pages/SparsityPage'
import { GraphPage } from './pages/GraphPage'
import { MonosemanticityPage } from './pages/MonosemanticityPage'
import { HebbianPage } from './pages/HebbianPage'
import { MergePage } from './pages/MergePage'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<HomePage />} />
          <Route path="architecture" element={<ArchitecturePage />} />
          <Route path="sparsity" element={<SparsityPage />} />
          <Route path="graph" element={<GraphPage />} />
          <Route path="monosemanticity" element={<MonosemanticityPage />} />
          <Route path="hebbian" element={<HebbianPage />} />
          <Route path="merge" element={<MergePage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
