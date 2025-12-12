import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { motion } from "framer-motion";

// LLMMarkdownViewer displays markdown text centered inside a transparent container
export default function LLMMarkdownViewer({ text, color = "#000", fontSizePx = 20, onRendered }) {
  return (
    <div className="w-full flex justify-center">
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        onAnimationComplete={onRendered}   // ← add this
        className="prose text-center bg-transparent w-full"
        style={{ width: "100%", color, fontSize: `${fontSizePx}px`, lineHeight: 1.5 }}
      >
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
      </motion.div>
    </div>
  );
}