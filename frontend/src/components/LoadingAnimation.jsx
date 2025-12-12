import React from "react"

export default function LoadingAnimation({ lang = 'en' }) {
  const color = '#ca1a21'
  const label = lang === 'pt' ? 'A processar o pedido' : 'Processing query'
  const circleStyle = (name, delayMs) => ({
    width: 16,
    height: 16,
    borderRadius: '50%',
    background: color,
    display: 'inline-block',
    margin: '0 8px',
    animation: `${name} 1200ms ease-in-out ${delayMs}ms infinite`
  })

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
      <style>
        {`
          /* Ball A: rise (0-25%), return (25-50%), hold (50-100%) */
          @keyframes cuoco-riseA {
            0% { transform: translateY(0); }
            25% { transform: translateY(-16px); }
            50% { transform: translateY(0); }
            100% { transform: translateY(0); }
          }
          /* Ball B: rise begins when A starts returning (25-50%), return (50-75%), hold (75-100%) */
          @keyframes cuoco-riseB {
            0% { transform: translateY(0); }
            25% { transform: translateY(0); }
            50% { transform: translateY(-16px); }
            75% { transform: translateY(0); }
            100% { transform: translateY(0); }
          }
        `}
      </style>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <span style={circleStyle('cuoco-riseA', 0)} />
        <span style={circleStyle('cuoco-riseB', 0)} />
      </div>
      <div style={{ marginTop: 10, color: '#000', fontFamily: 'Panton, sans-serif', fontSize: 22, fontWeight: 700 }}>
        {label}
      </div>
    </div>
  )
}
