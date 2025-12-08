// app/profile/components/CoverImage.tsx
"use client"
import React from "react"

export default function CoverImage({
  src,
  alt,
}: {
  src?: string
  alt?: string
}) {
  const cover = src ?? "/images/default-cover.jpg"
  return (
    <div className="w-full h-40 md:h-56 overflow-hidden rounded-sm mt-4">
      <img
        src={cover}
        alt={alt}
        className="w-full h-full object-cover rounded"
      />
    </div>
  )
}
