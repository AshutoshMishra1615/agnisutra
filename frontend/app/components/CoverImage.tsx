// app/profile/components/CoverImage.tsx
"use client";
import React from "react";

export default function CoverImage({
  src,
  alt,
}: {
  src?: string;
  alt?: string;
}) {
  const cover = src ?? "/images/default-cover.jpg";
  return (
    <div className="w-full h-40 md:h-64 overflow-hidden rounded-2xl mt-6 border border-[#879d7b]/20 relative group">
      <div className="absolute inset-0 bg-linear-to-t from-[#050b05] to-transparent opacity-60"></div>
      <img
        src={cover}
        alt={alt}
        className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105"
      />
    </div>
  );
}
