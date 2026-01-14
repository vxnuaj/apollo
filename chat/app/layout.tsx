import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Apollo - AI Chat",
  description: "Chat interface for LLM inference engine",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
