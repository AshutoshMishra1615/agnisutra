// app/profile/components/InfoRow.tsx
import React from "react"

export default function InfoRow({
  label,
  value,
  className = "",
}: {
  label: string
  value: React.ReactNode
  className?: string
}) {
  return (
    <div
      className={`flex flex-col md:flex-row md:items-center md:justify-between py-4 border-b border-white/10 ${className}`}
    >
      <div className="text-sm md:text-base font-semibold text-gray-200">
        {label}
      </div>
      <div className="mt-2 md:mt-0 text-sm md:text-base text-right text-gray-100">
        {value}
      </div>
    </div>
  )
}
