import * as React from 'react'

export type ModelType = 'gtsrb' | 'vn'

interface ModelTypeContextValue {
  modelType: ModelType
  setModelType: (mt: ModelType) => void
}

const ModelTypeContext = React.createContext<ModelTypeContextValue | null>(null)

const STORAGE_KEY = 'tsr-model-type'

export function ModelTypeProvider({ children }: { children: React.ReactNode }) {
  const [modelType, setModelTypeState] = React.useState<ModelType>(() => {
    const saved = localStorage.getItem(STORAGE_KEY)
    return saved === 'vn' || saved === 'gtsrb' ? saved : 'gtsrb'
  })

  const setModelType = React.useCallback((next: ModelType) => {
    setModelTypeState(next)
    localStorage.setItem(STORAGE_KEY, next)
  }, [])

  const value = React.useMemo(
    () => ({ modelType, setModelType }),
    [modelType, setModelType],
  )

  return (
    <ModelTypeContext.Provider value={value}>
      {children}
    </ModelTypeContext.Provider>
  )
}

export function useModelType(): ModelTypeContextValue {
  const ctx = React.useContext(ModelTypeContext)
  if (!ctx) throw new Error('useModelType must be used within ModelTypeProvider')
  return ctx
}
