export const CLASS_NAMES = ['Cat', 'Dog', 'Bird', 'Unknown'] as const

export type ClassName = (typeof CLASS_NAMES)[number]
export type DisplayClassName = 'Cat' | 'Dog' | 'Bird' | 'Other'

export const displayClassName = (name: ClassName): DisplayClassName =>
  name === 'Unknown' ? 'Other' : name
export type InputSource = 'upload' | 'url'
export type AppStatus = 'idle' | 'ready' | 'classifying' | 'success' | 'error'
export type ModelPhase = 'starting' | 'downloading' | 'optimizing' | 'ready' | 'error'
export type FeedbackChoice = 'correct' | 'incorrect'

export type ErrorCode =
  | 'invalid_type'
  | 'file_too_large'
  | 'invalid_url'
  | 'cors_blocked'
  | 'image_load_failed'
  | 'classification_failed'

export interface ValidationResult {
  ok: boolean
  errorCode?: ErrorCode
  message?: string
}

export interface PredictionResult {
  label: ClassName
  predictedIndex: number
  probabilities: [number, number, number, number]
  predictionMs: number
  timestamp: number
  backend: string
  modelLoadMs: number
  features: Float32Array
}

export interface ModelStatus {
  phase: ModelPhase
  progress: number | null
  backend: string | null
  message: string
}
