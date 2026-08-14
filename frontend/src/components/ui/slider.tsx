import * as React from 'react'
import { cn } from '@/lib/utils'

export interface SliderProps {
  value: number
  min?: number
  max?: number
  step?: number
  onValueChange?: (value: number) => void
  disabled?: boolean
  className?: string
  id?: string
  'aria-label'?: string
}

/** Lightweight single-value slider built on a native range input. */
const Slider = React.forwardRef<HTMLInputElement, SliderProps>(
  (
    { value, min = 0, max = 100, step = 1, onValueChange, className, ...props },
    ref,
  ) => (
    <input
      ref={ref}
      type="range"
      value={value}
      min={min}
      max={max}
      step={step}
      onChange={(e) => onValueChange?.(Number(e.target.value))}
      className={cn(
        'h-2 w-full cursor-pointer appearance-none rounded-full bg-secondary accent-primary disabled:cursor-not-allowed disabled:opacity-50',
        className,
      )}
      {...props}
    />
  ),
)
Slider.displayName = 'Slider'

export { Slider }
