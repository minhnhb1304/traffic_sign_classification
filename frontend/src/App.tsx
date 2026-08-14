import { createHashRouter, RouterProvider } from 'react-router-dom'
import { LanguageProvider } from '@/hooks/useLanguage'
import { ModelTypeProvider } from '@/hooks/useModelType'
import { AppShell } from '@/components/AppShell'
import { UploadPage } from '@/pages/UploadPage'
import { SnapshotPage } from '@/pages/SnapshotPage'
import { RealtimePage } from '@/pages/RealtimePage'

const router = createHashRouter([
  {
    path: '/',
    element: <AppShell />,
    children: [
      { index: true, element: <UploadPage /> },
      { path: 'snapshot', element: <SnapshotPage /> },
      { path: 'realtime', element: <RealtimePage /> },
    ],
  },
])

function App() {
  return (
    <LanguageProvider>
      <ModelTypeProvider>
        <RouterProvider router={router} />
      </ModelTypeProvider>
    </LanguageProvider>
  )
}

export default App
