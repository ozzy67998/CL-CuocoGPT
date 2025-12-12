import React from "react"

export default function LanguageSelector({ lang = 'en', onChange }) {
  const setLang = (value) => {
    if (onChange) onChange(value)
  }

  const selectedBg = '#ca1a21'
  const selectedColor = '#ffffff'
  const unselectedBg = '#ffffff'
  const unselectedColor = '#ca1a21'

  return (
    <div
      style={{
        position: 'relative',
        display: 'inline-flex',
        alignItems: 'center',
        background: '#ffffff',
        border: '1px solid #d1d5db',
        borderRadius: '9999px',
        boxShadow: '0 2px 6px rgba(0,0,0,0.08)',
        overflow: 'hidden',
        width: '80px',
        height: '36px'
      }}
    >
      {/* Sliding red indicator */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: lang === 'en' ? 0 : '44%',
          width: '56%',
          height: '100%',
          background: selectedBg,
          borderRadius: '9999px',
          transition: 'left 200ms ease'
        }}
      />

      <button
        type="button"
        onClick={() => setLang('en')}
        style={{
          width: '50%',
          height: '100%',
          boxSizing: 'border-box',
          padding: '0 12px',
          border: 'none',
          background: 'transparent',
          color: lang === 'en' ? selectedColor : unselectedColor,
          fontWeight: 800,
          cursor: 'pointer',
          zIndex: 1
        }}
      >EN</button>
      <button
        type="button"
        onClick={() => setLang('pt')}
        style={{
          width: '50%',
          height: '100%',
          boxSizing: 'border-box',
          padding: '0 0px',
          border: 'none',
          background: 'transparent',
          color: lang === 'pt' ? selectedColor : unselectedColor,
          fontWeight: 800,
          cursor: 'pointer',
          zIndex: 1
        }}
      >PT</button>
    </div>
  )
}
