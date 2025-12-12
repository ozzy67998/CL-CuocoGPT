import React, { useMemo, useState } from "react";
import cuoco from "../assets/cuoco.svg";

const phrasesByLang = {
  en: [
    "What are we cooking today?",
    "Ready to make something delicious?",
    "What’s in your kitchen right now?",
    "Need a recipe or cooking advice?",
    "What are you craving today?",
    "Breakfast, lunch, or dinner — what’s the plan?",
    "Do you want a quick meal or something fancy?",
    "Tell me what ingredients you’ve got!",
    "Looking for a recipe or a cooking tip?",
    "What dish are you in the mood for?",
    "Cooking time! What can I help you make?",
    "Are you cooking for one or for a crowd?",
    "Any dietary preferences I should know about?",
    "Sweet or savory — what are you thinking?",
  ],
  pt: [
    "O que vamos cozinhar hoje?",
    "Pronto para fazer algo delicioso?",
    "O que tens na cozinha agora?",
    "Queres uma receita ou uma dica de cozinha?",
    "Do que estás com vontade hoje?",
    "Pequeno-almoço, almoço ou jantar — qual é o plano?",
    "Queres algo rápido ou algo mais elaborado?",
    "Diz-me quais ingredientes tens!",
    "Estás à procura de uma receita ou de uma dica?",
    "Que prato estás com vontade de preparar?",
    "Hora de cozinhar! Em que posso ajudar?",
    "Vais cozinhar só para ti ou para mais pessoas?",
    "Alguma preferência ou restrição alimentar?",
    "Doce ou salgado — o que preferes?",
  ],
};

export default function WelcomeMessage({ lang = 'en', className = "", color = "#000", fontSize = 36 }) {
  const [fixedIndex] = useState(() => {
    const base = phrasesByLang.en.length;
    return Math.floor(Math.random() * base);
  })

  const phrase = useMemo(() => {
    const list = phrasesByLang[lang] || phrasesByLang.en;
    const idx = fixedIndex % list.length;
    return list[idx] || "";
  }, [lang, fixedIndex]);

  return (
    <div className={className} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0px' }}>
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: '10px' }}>
        <span style={{ fontSize: '56px', color: '#000' }}>
          <span style={{ fontFamily: 'Panton, sans-serif', fontWeight: 800 }}>Cuoco</span>
          <span style={{ fontFamily: 'Panton, sans-serif', fontWeight: 400 }}>GPT</span>
        </span>
        <img
          src={cuoco}
          alt="Cuoco"
          style={{ width: '300px', height: '300px', display: 'inline-block', transform: 'translateY(62px)' }}
        />
      </div>
      <p
        className={`text-center font-semibold`}
        style={{ color, fontSize: `${fontSize}px`, lineHeight: 1.2, transform: 'translateY(-30px)' }}
      >
        {phrase}
      </p>
    </div>
  );
}
