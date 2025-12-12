import React, { useEffect, useMemo, useState } from "react";

export default function CornerText({ lang = 'en', align = 'center', gapPx = 6, charDelayMs = 30, lineDelayMs = 0, startDelayMs = 500 }) {
  const lines = useMemo(() => {
    const firstLine = lang === 'pt' ? 'TRAZIDO POR' : 'POWERED BY'
    const secondLine = 'OSTRAQUINAS'
    const thirdLine = 'FCTUC | DEI'
    return [firstLine, secondLine, thirdLine]
  }, [lang])

  const [displayed, setDisplayed] = useState(['', '', ''])
  const [hasAnimated, setHasAnimated] = useState(false)

  useEffect(() => {
    let active = true
    const typeLine = async (lineIdx) => {
      const text = lines[lineIdx]
      for (let i = 0; i <= text.length && active; i++) {
        setDisplayed((prev) => {
          const next = [...prev]
          next[lineIdx] = text.slice(0, i)
          return next
        })
        await new Promise((r) => setTimeout(r, charDelayMs))
      }
      if (active) await new Promise((r) => setTimeout(r, lineDelayMs))
      if (active && lineIdx < lines.length - 1) {
        await typeLine(lineIdx + 1)
      } else if (active) {
        setHasAnimated(true)
      }
    }
    const start = async () => {
      if (startDelayMs > 0) {
        await new Promise((r) => setTimeout(r, startDelayMs))
      }
      if (active) await typeLine(0)
    }
    start()
    return () => { active = false }
    // Run only on initial mount to avoid retriggering on language change
  }, [])

  // On language change after initial animation, update text instantly without retyping
  useEffect(() => {
    if (hasAnimated) {
      setDisplayed([lines[0], lines[1], lines[2]])
    }
  }, [lang, hasAnimated, lines])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: align === 'center' ? 'center' : (align === 'right' ? 'flex-end' : 'flex-start'), gap: gapPx }}>
      {/* Reserve fixed height per line to avoid layout shift */}
      {/* Line 1: Panton Thin */}
      <div style={{ height: '38px', display: 'flex', alignItems: 'baseline' }}>
        <div style={{ fontFamily: 'Panton, sans-serif', fontWeight: 100, fontStyle: 'normal', fontSize: '32px', lineHeight: 1.2, color: '#000', whiteSpace: 'pre' }}>
          {displayed[0]}
        </div>
      </div>

      {/* Line 2: OS (Thin) + TRAQUINAS (ExtraBold), grey */}
      <div style={{ height: '48px', display: 'flex', alignItems: 'baseline' }}>
        <div style={{ fontSize: '40px', lineHeight: 1.2, color: '#6b7280', whiteSpace: 'pre' }}>
          {displayed[1].length < lines[1].length ? (
            (() => {
              const typed = displayed[1]
              const thinLen = 2 // "OS"
              const thinTyped = typed.slice(0, Math.min(thinLen, typed.length))
              const boldTyped = typed.slice(Math.min(thinLen, typed.length))
              return (
                <>
                  <span style={{ fontFamily: 'Panton, sans-serif', fontWeight: 100, fontStyle: 'normal' }}>{thinTyped}</span>
                  <span style={{ fontFamily: 'Panton, sans-serif', fontWeight: 800, fontStyle: 'normal' }}>{boldTyped}</span>
                </>
              )
            })()
          ) : (
            <>
              <span style={{ fontFamily: 'Panton, sans-serif', fontWeight: 100, fontStyle: 'normal' }}>OS</span>
              <span style={{ fontFamily: 'Panton, sans-serif', fontWeight: 800, fontStyle: 'normal' }}>TRAQUINAS</span>
            </>
          )}
        </div>
      </div>

      {/* Line 3: FCTUC (Bold while typing) + " | DEI" (Regular) */}
      <div style={{ height: '44px', display: 'flex', alignItems: 'baseline' }}>
        <div style={{ fontSize: '36px', lineHeight: 1.2, color: '#000', whiteSpace: 'pre' }}>
          {/* While typing, render bold for "DEI" portion and regular for the rest */}
          {displayed[2].length < lines[2].length ? (
            (() => {
              const typed = displayed[2];
              const boldLen = 5; // "FCTUC" length
              const boldTyped = typed.slice(0, Math.min(boldLen, typed.length));
              const restTyped = typed.slice(Math.min(boldLen, typed.length));
              return (
                <>
                  <span style={{ fontFamily: 'Panton, sans-serif', fontWeight: 700, fontStyle: 'normal', color: '#000' }}>{boldTyped}</span>
                  <span style={{ fontFamily: 'Panton, sans-serif', fontWeight: 400, fontStyle: 'normal', color: '#000' }}>{restTyped}</span>
                </>
              );
            })()
          ) : (
            <>
              <span style={{ fontFamily: 'Panton, sans-serif', fontWeight: 700, fontStyle: 'normal', color: '#000' }}>FCTUC</span>
              <span style={{ fontFamily: 'Panton, sans-serif', fontWeight: 400, fontStyle: 'normal', color: '#000' }}> | DEI</span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
