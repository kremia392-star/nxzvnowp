import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

// Merge Tailwind classes
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Format numbers with commas
export function formatNumber(num: number): string {
  return num.toLocaleString()
}

// Format percentage
export function formatPercent(num: number, decimals = 1): string {
  return `${(num * 100).toFixed(decimals)}%`
}

// Format bytes to human readable
export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`
}

// Debounce function
export function debounce<T extends (...args: unknown[]) => unknown>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout>
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId)
    timeoutId = setTimeout(() => fn(...args), delay)
  }
}

// Throttle function
export function throttle<T extends (...args: unknown[]) => unknown>(
  fn: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle = false
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      fn(...args)
      inThrottle = true
      setTimeout(() => (inThrottle = false), limit)
    }
  }
}

// Generate color from string (for consistent colors)
export function stringToColor(str: string): string {
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash)
  }
  const hue = hash % 360
  return `hsl(${hue}, 70%, 60%)`
}

// Interpolate between two colors
export function interpolateColor(color1: string, color2: string, factor: number): string {
  const hex = (c: string) => parseInt(c, 16)
  const r1 = hex(color1.slice(1, 3))
  const g1 = hex(color1.slice(3, 5))
  const b1 = hex(color1.slice(5, 7))
  const r2 = hex(color2.slice(1, 3))
  const g2 = hex(color2.slice(3, 5))
  const b2 = hex(color2.slice(5, 7))
  
  const r = Math.round(r1 + (r2 - r1) * factor)
  const g = Math.round(g1 + (g2 - g1) * factor)
  const b = Math.round(b1 + (b2 - b1) * factor)
  
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`
}

// Decode UTF-8 bytes to string
export function bytesToString(bytes: number[]): string {
  return new TextDecoder().decode(new Uint8Array(bytes))
}

// Encode string to UTF-8 bytes
export function stringToBytes(str: string): number[] {
  return Array.from(new TextEncoder().encode(str))
}

// Safe character display (for non-printable chars)
export function safeChar(byte: number): string {
  if (byte >= 32 && byte < 127) {
    return String.fromCharCode(byte)
  }
  if (byte === 32) return '␣'
  if (byte === 10) return '↵'
  if (byte === 9) return '→'
  return `\\x${byte.toString(16).padStart(2, '0')}`
}
