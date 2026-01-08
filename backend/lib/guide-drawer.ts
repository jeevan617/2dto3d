export function drawGuide(ctx: CanvasRenderingContext2D, objectName: string, width: number, height: number, color: string = "rgba(0, 0, 0, 0.15)") {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5; // Thinner, finer lines
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();

    const cx = width / 2;
    const cy = height / 2;

    // Scale to 0.95x - Large but safe (fits inside)
    ctx.save();
    ctx.translate(cx, cy);
    ctx.scale(0.95, 0.95);
    ctx.translate(-cx, -cy);

    switch (objectName.toLowerCase()) {
        case 'ac':
            // Wide rectangle with a slat
            const acW = width * 0.7;
            const acH = height * 0.25;
            ctx.strokeRect(cx - acW / 2, cy - acH / 2, acW, acH);
            ctx.moveTo(cx - acW / 2, cy + acH / 6);
            ctx.lineTo(cx + acW / 2, cy + acH / 6);
            break;

        case 'fan':
            // Ceiling fan: center circle + 3 blades
            const r = width * 0.05;
            ctx.arc(cx, cy, r, 0, Math.PI * 2);
            for (let i = 0; i < 3; i++) {
                const angle = (Math.PI * 2 * i) / 3 - Math.PI / 2;
                ctx.moveTo(cx + Math.cos(angle) * r, cy + Math.sin(angle) * r);
                const ex = cx + Math.cos(angle) * (width * 0.35);
                const ey = cy + Math.sin(angle) * (width * 0.35);
                ctx.quadraticCurveTo(cx + Math.cos(angle + 0.5) * (width * 0.2), cy + Math.sin(angle + 0.5) * (width * 0.2), ex, ey);
                ctx.quadraticCurveTo(cx + Math.cos(angle - 0.5) * (width * 0.2), cy + Math.sin(angle - 0.5) * (width * 0.2), cx + Math.cos(angle) * r, cy + Math.sin(angle) * r);
            }
            break;

        case 'fridge':
            // Tall rectangle, split into freezer/main
            const frW = width * 0.35;
            const frH = height * 0.7;
            ctx.strokeRect(cx - frW / 2, cy - frH / 2, frW, frH);
            ctx.moveTo(cx - frW / 2, cy - frH / 2 + frH * 0.25);
            ctx.lineTo(cx + frW / 2, cy - frH / 2 + frH * 0.25); // Split
            ctx.moveTo(cx - frW / 2 + 10, cy - frH / 2 + frH * 0.1);
            ctx.lineTo(cx - frW / 2 + 10, cy - frH / 2 + frH * 0.2); // Handle
            break;

        case 'tv':
            // Wide screen + stand
            const tvW = width * 0.6;
            const tvH = width * 0.35;
            ctx.strokeRect(cx - tvW / 2, cy - tvH / 2, tvW, tvH);
            ctx.moveTo(cx, cy + tvH / 2);
            ctx.lineTo(cx, cy + tvH / 2 + 20); // Stem
            ctx.moveTo(cx - 20, cy + tvH / 2 + 20);
            ctx.lineTo(cx + 20, cy + tvH / 2 + 20); // Base
            break;

        case 'washmachine':
            // Square box + circle window
            const wmS = width * 0.5;
            ctx.strokeRect(cx - wmS / 2, cy - wmS / 2, wmS, wmS);
            ctx.moveTo(cx + wmS * 0.3, cy);
            ctx.arc(cx, cy, wmS * 0.25, 0, Math.PI * 2);
            ctx.moveTo(cx - wmS / 2, cy - wmS / 2 + wmS * 0.15); // Control panel
            ctx.lineTo(cx + wmS / 2, cy - wmS / 2 + wmS * 0.15);
            break;

        // --- Furniture ---
        case 'chair':
            const chW = width * 0.3;
            const chH = height * 0.3;
            ctx.strokeRect(cx - chW / 2, cy - chH / 2, chW, chH); // Back
            ctx.strokeRect(cx - chW / 2, cy + chH / 2, chW, 10); // Seat
            ctx.moveTo(cx - chW / 2, cy + chH / 2 + 10); ctx.lineTo(cx - chW / 2, cy + chH / 2 + 60); // FL
            ctx.moveTo(cx + chW / 2, cy + chH / 2 + 10); ctx.lineTo(cx + chW / 2, cy + chH / 2 + 60); // FR
            break;

        case 'table':
            ctx.strokeRect(cx - width * 0.4, cy - 10, width * 0.8, 20); // Top
            ctx.moveTo(cx - width * 0.35, cy + 10); ctx.lineTo(cx - width * 0.35, cy + 70); // Leg L
            ctx.moveTo(cx + width * 0.35, cy + 10); ctx.lineTo(cx + width * 0.35, cy + 70); // Leg R
            break;

        case 'bed':
            ctx.strokeRect(cx - width * 0.4, cy, width * 0.8, height * 0.15); // Mattress
            ctx.strokeRect(cx - width * 0.4, cy - 20, width * 0.8, 20); // Headboard
            ctx.moveTo(cx - width * 0.35, cy + height * 0.15); ctx.lineTo(cx - width * 0.35, cy + height * 0.15 + 20); // Leg
            ctx.moveTo(cx + width * 0.35, cy + height * 0.15); ctx.lineTo(cx + width * 0.35, cy + height * 0.15 + 20); // Leg
            break;

        case 'sofa':
            ctx.strokeRect(cx - width * 0.4, cy, width * 0.8, height * 0.2); // Base
            ctx.strokeRect(cx - width * 0.4, cy - height * 0.15, width * 0.8, height * 0.15); // Back
            ctx.strokeRect(cx - width * 0.45, cy - height * 0.05, width * 0.05, height * 0.25); // Arm L
            ctx.strokeRect(cx + width * 0.4, cy - height * 0.05, width * 0.05, height * 0.25); // Arm R
            break;

        case 'cupboard':
            ctx.strokeRect(cx - width * 0.25, cy - height * 0.35, width * 0.5, height * 0.7); // Body
            ctx.moveTo(cx, cy - height * 0.35); ctx.lineTo(cx, cy + height * 0.35); // Split
            ctx.moveTo(cx - 5, cy); ctx.lineTo(cx - 5, cy + 15); // Handle L
            ctx.moveTo(cx + 5, cy); ctx.lineTo(cx + 5, cy + 15); // Handle R
            break;

        // --- Gadgets (Refined for Size) ---
        case 'mobile':
            ctx.strokeRect(cx - width * 0.25, cy - height * 0.4, width * 0.5, height * 0.8); // Body
            ctx.strokeRect(cx - width * 0.22, cy - height * 0.35, width * 0.44, height * 0.65); // Screen
            ctx.beginPath(); ctx.arc(cx, cy + height * 0.35, width * 0.05, 0, Math.PI * 2); ctx.stroke(); // Button
            break;

        case 'tablet':
            ctx.strokeRect(cx - width * 0.4, cy - height * 0.28, width * 0.8, height * 0.56); // Body (Lands)
            ctx.strokeRect(cx - width * 0.36, cy - height * 0.24, width * 0.72, height * 0.48); // Screen
            break;

        case 'laptop':
            ctx.strokeRect(cx - width * 0.4, cy - height * 0.3, width * 0.8, height * 0.45); // Screen
            ctx.strokeRect(cx - width * 0.45, cy + height * 0.15, width * 0.9, 20); // Base
            break;

        case 'headphones':
            ctx.beginPath(); ctx.arc(cx, cy, width * 0.3, Math.PI, 0); ctx.stroke(); // Band
            ctx.strokeRect(cx - width * 0.35, cy - 10, width * 0.1, height * 0.25); // Cup L
            ctx.strokeRect(cx + width * 0.25, cy - 10, width * 0.1, height * 0.25); // Cup R
            break;

        case 'watch':
            const wR = width * 0.2; // Much bigger radius
            ctx.beginPath(); ctx.arc(cx, cy, wR, 0, Math.PI * 2); ctx.stroke(); // Face
            ctx.strokeRect(cx - wR * 0.4, cy - wR * 2.5, wR * 0.8, wR * 1.5); // Strap Top
            ctx.strokeRect(cx - wR * 0.4, cy + wR, wR * 0.8, wR * 1.5); // Strap Bot
            break;

        // --- Kitchen (Refined for Size) ---
        case 'cooker':
            const ckW = width * 0.5;
            const ckH = height * 0.4;
            ctx.strokeRect(cx - ckW / 2, cy, ckW, ckH); // Pot
            ctx.beginPath(); ctx.moveTo(cx - ckW / 2, cy); ctx.quadraticCurveTo(cx, cy - ckH * 0.5, cx + ckW / 2, cy); ctx.stroke(); // Lid
            ctx.moveTo(cx, cy - ckH * 0.2); ctx.lineTo(cx, cy - ckH * 0.4); // Whistle
            ctx.moveTo(cx - ckW * 0.6, cy + ckH * 0.2); ctx.lineTo(cx - ckW * 0.5, cy + ckH * 0.2); // Handle
            break;

        case 'stove':
            ctx.strokeRect(cx - width * 0.4, cy, width * 0.8, 40); // Base
            ctx.beginPath(); ctx.arc(cx - width * 0.2, cy, width * 0.1, 0, Math.PI * 2); ctx.stroke(); // Burner L
            ctx.beginPath(); ctx.arc(cx + width * 0.2, cy, width * 0.1, 0, Math.PI * 2); ctx.stroke(); // Burner R
            break;

        case 'oven':
            ctx.strokeRect(cx - width * 0.4, cy - height * 0.3, width * 0.8, height * 0.6);
            ctx.strokeRect(cx - width * 0.3, cy - height * 0.2, width * 0.6, height * 0.4); // Glass
            break;

        case 'mixer':
            ctx.strokeRect(cx - width * 0.15, cy, width * 0.3, height * 0.3); // Base
            ctx.moveTo(cx - width * 0.15, cy); ctx.lineTo(cx - width * 0.2, cy - height * 0.4);
            ctx.lineTo(cx + width * 0.2, cy - height * 0.4); ctx.lineTo(cx + width * 0.15, cy); // Jar
            break;

        case 'tostar':
            ctx.strokeRect(cx - width * 0.3, cy - height * 0.15, width * 0.6, height * 0.3);
            ctx.moveTo(cx - width * 0.15, cy - height * 0.15); ctx.lineTo(cx - width * 0.15, cy - height * 0.25);
            ctx.moveTo(cx + width * 0.15, cy - height * 0.15); ctx.lineTo(cx + width * 0.15, cy - height * 0.25);
            break;

        // --- Sports (Refined for Size) ---
        case 'football':
        case 'basketball':
            ctx.beginPath(); ctx.arc(cx, cy, width * 0.35, 0, Math.PI * 2); ctx.stroke();
            ctx.beginPath(); ctx.arc(cx, cy, width * 0.25, 0, Math.PI * 2); ctx.stroke(); // Detail
            break;

        case 'tennisball':
            ctx.beginPath(); ctx.arc(cx, cy, width * 0.25, 0, Math.PI * 2); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(cx - width * 0.25, cy); ctx.bezierCurveTo(cx - width * 0.1, cy - width * 0.15, cx + width * 0.1, cy + width * 0.15, cx + width * 0.25, cy); ctx.stroke();
            break;

        case 'cricketbat':
            ctx.strokeRect(cx - width * 0.1, cy - height * 0.2, width * 0.2, height * 0.6); // Blade
            ctx.strokeRect(cx - width * 0.03, cy - height * 0.4, width * 0.06, height * 0.2); // Handle
            break;

        case 'hockey':
            ctx.beginPath();
            ctx.moveTo(cx + width * 0.1, cy - height * 0.4);
            ctx.lineTo(cx + width * 0.1, cy + height * 0.2);
            ctx.quadraticCurveTo(cx + width * 0.1, cy + height * 0.4, cx - width * 0.2, cy + height * 0.4);
            ctx.stroke();
            break;

        case 'racket':
            ctx.beginPath(); ctx.ellipse(cx, cy - height * 0.1, width * 0.25, height * 0.3, 0, 0, Math.PI * 2); ctx.stroke(); // Head
            ctx.moveTo(cx, cy + height * 0.2); ctx.lineTo(cx, cy + height * 0.5); // Handle
            break;

        // --- Fashion ---
        case 'shirt':
            ctx.moveTo(cx, cy - height * 0.3); ctx.lineTo(cx + width * 0.3, cy - height * 0.3); ctx.lineTo(cx + width * 0.4, cy - height * 0.1); // R shoulder
            ctx.moveTo(cx, cy - height * 0.3); ctx.lineTo(cx - width * 0.3, cy - height * 0.3); ctx.lineTo(cx - width * 0.4, cy - height * 0.1); // L shoulder
            ctx.moveTo(cx - width * 0.3, cy - height * 0.3); ctx.lineTo(cx - width * 0.3, cy + height * 0.4); ctx.lineTo(cx + width * 0.3, cy + height * 0.4); ctx.lineTo(cx + width * 0.3, cy - height * 0.3); // Body
            break;

        case 'bag':
            ctx.strokeRect(cx - width * 0.25, cy - height * 0.2, width * 0.5, height * 0.4); // Body
            ctx.beginPath(); ctx.arc(cx, cy - height * 0.2, width * 0.15, Math.PI, 0); ctx.stroke(); // Handle
            break;

        case 'cap':
            ctx.beginPath(); ctx.arc(cx, cy, width * 0.25, Math.PI, 0); ctx.stroke(); // Dome
            ctx.strokeRect(cx - width * 0.3, cy, width * 0.6, 10); // Rim
            ctx.strokeRect(cx + width * 0.3, cy, width * 0.15, 10); // Visor
            break;

        case 'shoe':
            const sW = width * 0.6;
            ctx.beginPath();
            ctx.moveTo(cx - sW / 2, cy + 20); // Heel
            ctx.lineTo(cx - sW / 2, cy + 60);
            ctx.lineTo(cx + sW / 2, cy + 60); // Sole
            ctx.quadraticCurveTo(cx + sW / 2 + 20, cy + 60, cx + sW / 2 + 20, cy + 40); // Toe
            ctx.lineTo(cx + sW / 4, cy - 20);
            ctx.lineTo(cx - sW / 2, cy + 20);
            ctx.stroke();
            break;

        // --- Instruments ---
        case 'guiter':
            ctx.beginPath();
            ctx.ellipse(cx, cy + height * 0.15, width * 0.2, height * 0.25, 0, 0, Math.PI * 2); ctx.stroke(); // Body
            ctx.strokeRect(cx - width * 0.04, cy - height * 0.4, width * 0.08, height * 0.55); // Neck
            break;

        case 'drum':
            const dW = width * 0.5;
            ctx.beginPath(); ctx.ellipse(cx, cy - height * 0.15, dW / 2, height * 0.1, 0, 0, Math.PI * 2); ctx.stroke(); // Top
            ctx.moveTo(cx - dW / 2, cy - height * 0.15); ctx.lineTo(cx - dW / 2, cy + height * 0.15); // Side
            ctx.moveTo(cx + dW / 2, cy - height * 0.15); ctx.lineTo(cx + dW / 2, cy + height * 0.15); // Side
            ctx.beginPath(); ctx.ellipse(cx, cy + height * 0.15, dW / 2, height * 0.1, 0, 0, Math.PI, false); ctx.stroke(); // Bot
            break;

        case 'flute':
            ctx.strokeRect(cx - width * 0.4, cy - 10, width * 0.8, 20);
            ctx.beginPath(); ctx.arc(cx - width * 0.3, cy, 6, 0, Math.PI * 2); ctx.stroke();
            break;

        case 'piano':
            ctx.strokeRect(cx - width * 0.35, cy, width * 0.7, height * 0.2); // Keys
            ctx.strokeRect(cx - width * 0.35, cy - height * 0.2, width * 0.7, height * 0.2); // Body
            break;

        // --- Vehicles ---
        case 'car':
            const cw = width * 0.7;
            ctx.moveTo(cx - cw / 2, cy + 10); ctx.lineTo(cx + cw / 2, cy + 10); // Base
            ctx.moveTo(cx - cw / 3, cy + 10); ctx.lineTo(cx - cw / 4, cy - 40); ctx.lineTo(cx + cw / 4, cy - 40); ctx.lineTo(cx + cw / 3, cy + 10); // Roof
            ctx.beginPath(); ctx.arc(cx - cw / 4, cy + 30, 20, 0, Math.PI * 2); ctx.stroke(); // Wheel L
            ctx.beginPath(); ctx.arc(cx + cw / 4, cy + 30, 20, 0, Math.PI * 2); ctx.stroke(); // Wheel R
            break;

        case 'bus':
            ctx.strokeRect(cx - width * 0.35, cy - height * 0.2, width * 0.7, height * 0.4); // Body
            ctx.beginPath(); ctx.arc(cx - width * 0.2, cy + height * 0.2, 20, 0, Math.PI * 2); ctx.stroke(); // Wheel L
            ctx.beginPath(); ctx.arc(cx + width * 0.2, cy + height * 0.2, 20, 0, Math.PI * 2); ctx.stroke(); // Wheel R
            // Windows
            ctx.strokeRect(cx - width * 0.3, cy - height * 0.15, width * 0.15, height * 0.15);
            ctx.strokeRect(cx - width * 0.05, cy - height * 0.15, width * 0.15, height * 0.15);
            ctx.strokeRect(cx + width * 0.2, cy - height * 0.15, width * 0.1, height * 0.15);
            break;

        case 'aeroplane':
            ctx.moveTo(cx - width * 0.4, cy); ctx.lineTo(cx + width * 0.4, cy); ctx.stroke(); // Body
            ctx.moveTo(cx - width * 0.1, cy); ctx.lineTo(cx - width * 0.2, cy - height * 0.3); ctx.stroke(); // Wing Top
            ctx.moveTo(cx - width * 0.1, cy); ctx.lineTo(cx - width * 0.2, cy + height * 0.3); ctx.stroke(); // Wing Bot
            ctx.moveTo(cx + width * 0.3, cy); ctx.lineTo(cx + width * 0.4, cy - height * 0.1); ctx.stroke(); // Tail
            break;

        case 'bike':
            ctx.beginPath(); ctx.arc(cx - width * 0.25, cy + 20, width * 0.12, 0, Math.PI * 2); ctx.stroke(); // Wheel L
            ctx.beginPath(); ctx.arc(cx + width * 0.25, cy + 20, width * 0.12, 0, Math.PI * 2); ctx.stroke(); // Wheel R
            ctx.moveTo(cx - width * 0.25, cy + 20); ctx.lineTo(cx, cy - 20); ctx.lineTo(cx + width * 0.25, cy + 20); // Frame
            ctx.moveTo(cx, cy - 20); ctx.lineTo(cx - 20, cy - 50); // Handle
            break;

        // --- Stationary ---
        case 'pen':
        case 'pencil':
            const pL = width * 0.7;
            ctx.moveTo(cx - pL / 2, cy); ctx.lineTo(cx + pL / 4, cy); // Shaft
            ctx.lineTo(cx + pL / 2, cy + 10); ctx.lineTo(cx + pL / 4, cy + 20); // Tip
            ctx.lineTo(cx - pL / 2, cy + 20); ctx.lineTo(cx - pL / 2, cy); // Close
            break;

        case 'ruler':
            const rW = width * 0.8;
            ctx.strokeRect(cx - rW / 2, cy - 20, rW, 40);
            // Ticks
            for (let i = 0; i < 15; i++) {
                ctx.moveTo(cx - rW / 2 + 20 + i * 20, cy - 20);
                ctx.lineTo(cx - rW / 2 + 20 + i * 20, cy - 10);
            }
            break;

        case 'book':
            ctx.strokeRect(cx - width * 0.25, cy - height * 0.3, width * 0.5, height * 0.6); // Cover
            ctx.moveTo(cx - width * 0.2, cy - height * 0.3); ctx.lineTo(cx - width * 0.2, cy + height * 0.3); // Spine
            break;

        case 'easer':
            const eW = width * 0.4;
            ctx.moveTo(cx - eW / 2, cy - 20); ctx.lineTo(cx + 20, cy - 40); ctx.lineTo(cx + eW / 2, cy - 20); // Top
            ctx.lineTo(cx + eW / 2, cy + 20); ctx.lineTo(cx - 20, cy + 40); ctx.lineTo(cx - eW / 2, cy + 20); ctx.lineTo(cx - eW / 2, cy - 20); // Sides
            break;

        default:
            // Generic Fallback
            ctx.setLineDash([5, 5]);
            ctx.strokeRect(cx - width * 0.25, cy - height * 0.25, width * 0.5, height * 0.5);
            ctx.setLineDash([]);
            break;
    }

    ctx.stroke();
}
