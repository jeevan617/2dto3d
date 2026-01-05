import React, { useState, useEffect } from 'react';
import './AnimatedGreeting.css';

const greetings = [
  'everyone',        // English
  'सभी लोग',         // Hindi
  'todos',           // Spanish
  'tout le monde',   // French
  'みんな',           // Japanese
  '여러분'            // Korean
];

const AnimatedGreeting: React.FC = () => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [fade, setFade] = useState(true);

  useEffect(() => {
    const interval = setInterval(() => {
      setFade(false);
      setTimeout(() => {
        setCurrentIndex((prev) => (prev + 1) % greetings.length);
        setFade(true);
      }, 800);
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="animated-greet-container">
      <div className="greet">
        <span className="static-part">Hii </span>
        <span className={`changing-part ${fade ? 'fade-in' : 'fade-out'}`}>
          {greetings[currentIndex]}
        </span>
      </div>
    </div>
  );
};

export default AnimatedGreeting;