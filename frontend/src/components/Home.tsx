// src/components/Home.tsx
import React from 'react';
import Spline from '@splinetool/react-spline';
import AnimatedGreeting from './AnimatedGreeting';
import MusicButton from './MusicButton';
import './Home.css';

const Home: React.FC = () => {
  const handleRedirect = () => {
    window.location.href = 'http://localhost:5005';
  };
  return (
    <div className="home-section">

      {/* Left half: Robot + greeting overlay */}
      <div className="robot-half">
        <Spline scene="/scene.splinecode" />
        <div className="greeting-overlay">
          <AnimatedGreeting />
        </div>
      </div>

      {/* Right half: Spline background + CTA overlay */}
      <div className="right-half">
        {/* Spline scene as background */}
        <Spline scene="/scene2.splinecode" className="spline-bg" />

        {/* CTA overlay on top */}
        <div className="cta-overlay">
          <div className="engine-badge">SK3D-v2.1 Stable (CUDA Core)</div>
          <h2>Design your 3D & 2D</h2>
          <div className="cta-buttons">
            <button onClick={handleRedirect}>Sketch to 2D & 3D</button>

            {/* Music button */}
            <MusicButton src="/music.mp3" />
          </div>
        </div>
      </div>

      <div className="tech-footer">
        <span className="tech-item">Inference: 42ms</span>
        <span className="spacer">|</span>
        <span className="tech-item">Precision: FP16</span>
        <span className="spacer">|</span>
        <span className="tech-item">Weights: sk2d_latest.pth</span>
      </div>

    </div>
  );
};

export default Home;