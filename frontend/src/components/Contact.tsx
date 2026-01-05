import React from 'react';
import './Contact.css';

const Contact: React.FC = () => {
  return (
    <section id="contact" className="contact-section">
      <div className="contact-container">
        <div className="contact-header">
          <h2>Get in Touch</h2>
          <p className="subtitle">Have a question or want to collaborate? Reach out to the SK3D Team.</p>
        </div>

        <div className="contact-grid">
          {/* Left Column: Info */}
          <div className="contact-info">
            <div className="info-card">
              <h3>Contact Information</h3>
              <p>Fill out the form and our team will get back to you within 24 hours.</p>

              <div className="info-details">
                <div className="info-item">
                  <span className="icon">📧</span>
                  <p>support@sketchers.io</p>
                </div>
                <div className="info-item">
                  <span className="icon">📞</span>
                  <p>+91 123 456 7890</p>
                </div>
                <div className="info-item">
                  <span className="icon">📍</span>
                  <p>jeevan M and team, Bangalore, India</p>
                </div>
              </div>

              <div className="social-links">
                <span className="social-icon">𝕏</span>
                <span className="social-icon">💼</span>
                <span className="social-icon">🐙</span>
              </div>
            </div>
          </div>

          {/* Right Column: Form */}
          <div className="contact-form-container">
            <form className="contact-form" onSubmit={(e) => e.preventDefault()}>
              <div className="form-group">
                <label>Your Name</label>
                <input type="text" placeholder="Jeevan M" required />
              </div>
              <div className="form-group">
                <label>Email Address</label>
                <input type="email" placeholder="jeevan@example.com" required />
              </div>
              <div className="form-group">
                <label>Subject</label>
                <select>
                  <option>Technical Support</option>
                  <option>Business Inquiry</option>
                  <option>Feedback</option>
                  <option>Other</option>
                </select>
              </div>
              <div className="form-group">
                <label>Message</label>
                <textarea placeholder="Tell us more about your project..." rows={4} required></textarea>
              </div>
              <button type="submit" className="submit-btn">Send Message</button>
            </form>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Contact;