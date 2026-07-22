import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import DeviceGate from './DeviceGate'
import './styles/gate.css'

createRoot(document.getElementById('root')!).render(<StrictMode><DeviceGate /></StrictMode>)
