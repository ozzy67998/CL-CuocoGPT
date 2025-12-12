import React from "react"
import { motion } from "framer-motion"

export default function UserMessageBubble({ text, maxWidthPx = 566 }) {
  const containerStyle = {
    display: "flex",
    justifyContent: "flex-end",
    width: "100%",
  }

  const bubbleStyle = {
    background: "#ffffff",
    color: "#222",
    padding: "12px 18px",
    borderRadius: "20px",
    boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
    fontSize: "18px",
    lineHeight: 1.4,
    maxWidth: `${maxWidthPx}px`,
    width: "fit-content",
    wordBreak: "break-word",
  }

  return (
    <div style={containerStyle}>
      <motion.div
        style={bubbleStyle}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        {text}
      </motion.div>
    </div>
  )
}
