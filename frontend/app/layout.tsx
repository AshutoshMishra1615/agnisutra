import { NextIntlClientProvider } from "next-intl";
import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { getLocale, getMessages } from "next-intl/server";
import axios from "axios";
import { Toaster } from "sonner";
import { AuthProvider } from "./context/AuthContext";


axios.defaults.baseURL = process.env.BACKEND_URL;

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "AgniSutra",
  description: "Smart Farm Management",
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const locale = await getLocale();
  const messages = await getMessages();

  return (
    <html lang={locale}>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <NextIntlClientProvider locale={locale} messages={messages}>
          <Toaster/>
          <AuthProvider>
              {children}
          </AuthProvider>
        </NextIntlClientProvider>
      </body>
    </html>
  );
}
