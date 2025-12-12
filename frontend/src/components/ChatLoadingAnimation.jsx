import React from "react"
import { motion } from "framer-motion"

export default function ChatLoadingAnimation() {
  const dotStyle = {
    width: 10,
    height: 10,
    backgroundColor: "#ca1a21",
    borderRadius: "50%",
    display: "inline-block",
    margin: "0 4px",
  }

  return (
    <div style={{ padding: "20px 0", display: "flex", alignItems: "center" }}>
      {[0, 1].map((i) => (
        <motion.span
          key={i}
          style={dotStyle}
          animate={{ y: [0, -8, 0] }}
          transition={{
            duration: 0.8,
            repeat: Infinity,
            repeatDelay: 0.4,
            ease: "easeInOut",
            delay: i * 0.2,
          }}
        />
      ))}
    </div>
  )
}
