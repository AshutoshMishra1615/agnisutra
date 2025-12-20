// app/profile/components/InfoRow.tsx
import React from "react";

export default function InfoRow({
  label,
  value,
  className = "",
}: {
  label: string;
  value: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`flex flex-col md:flex-row md:items-center md:justify-between py-4 border-b border-[#879d7b]/10 last:border-0 ${className}`}
    >
      <div className="text-xs md:text-sm font-bold text-[#879d7b] tracking-wider">
        {label}
      </div>
      <div className="mt-1 md:mt-0 text-sm md:text-base text-white font-medium">
        {value}
      </div>
    </div>
  );
}
