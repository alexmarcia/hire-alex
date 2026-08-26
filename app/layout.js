import "./globals.css";

export const metadata = {
  title: "Hire Alex Marcia-Gonzalez | Data Science & AI Engineering",
  description:
    "Chat with Alex about his data science, LLM engineering, and FAA/DoD cost analysis background. Paste a job description to check fit.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
