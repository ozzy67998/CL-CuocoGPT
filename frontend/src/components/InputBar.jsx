import React, { forwardRef } from "react";
import NewChatButton from "./NewChatButton.jsx";

const InputBar = forwardRef(({ placeholder, value, onChange, onKeyDown, onNewChat }, ref) => {
  return (
    <div style={{ position: "relative", width: "100%" }}>
      <input
        ref={ref}
        type="text"
        placeholder={placeholder}
        value={value}
        onChange={onChange}
        onKeyDown={onKeyDown}
        style={{
          border: "2px solid #d3d3d3ff",
          borderRadius: "25px",
          padding: "16px 24px",
          paddingRight: "104px",
          outline: "none",
          fontSize: "24px",
          width: "100%",
          transition: "border-color 0.2s",
          boxSizing: "border-box",
          background: "#ffffff",
        }}
        onFocus={(e) => (e.target.style.borderColor = "#a1a1a1ff")}
        onBlur={(e) => (e.target.style.borderColor = "#c0c0c0ff")}
      />

      <div
        style={{
          position: "absolute",
          right: "0px",
          top: "50%",
          transform: "translateY(-50%)",
        }}
      >
        <NewChatButton onClick={onNewChat} />
      </div>
    </div>
  );
});

export default InputBar;

