export default function NewChatButton({ onClick, title = "New chat" }) {
  return (
    <button
      type="button"
      aria-label={title}
      title={title}
      onClick={onClick}
      style={{
        width: "64px",
        height: "64px",
        borderRadius: "25px",
        border: "none",
        background: "transparent",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: "36px",
        lineHeight: 1,
        cursor: "pointer",
        color: "#333",
        transition: "box-shadow 0.2s ease, color 0.2s ease",
        boxShadow: "inset 0 0 0 0 rgba(229,57,53,0)",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.boxShadow = "inset 0 0 0 999px #ca1a21"; // fill to edges
        e.currentTarget.style.color = "#ffffff"; // white '+' symbol
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.boxShadow = "inset 0 0 0 0 rgba(229,57,53,0)";
        e.currentTarget.style.color = "#333";
      }}
    >
      +
    </button>
  );
}
