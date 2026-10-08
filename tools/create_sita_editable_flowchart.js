const pptxgen = require('/Users/ming/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'Codex';
pptx.company = 'OpenAI';
pptx.subject = 'Editable reconstruction of the SITA flowchart';
pptx.title = 'SITA: relevance-weighted source initialization with target-only inference';
pptx.lang = 'en-US';

const slide = pptx.addSlide();
slide.background = { color: 'FFFFFF' };

const ST = pptx.ShapeType;
const SCALE = 13.333 / 2048;
const px = (x, y, w, h) => ({ x: x * SCALE, y: y * SCALE, w: w * SCALE, h: h * SCALE });

const FONT = 'Times New Roman';
const MATH = 'Cambria Math';
const UI = 'Arial';
const BLACK = '111111';
const MID = '333333';
const PALE_YELLOW = 'FFF1C9';
const PALE_GREEN = 'EFF9E9';
const GREEN = '7FBD85';
const PALE_BLUE = 'E3F2FF';
const BLUE = '8AB5D6';
const LIGHT_GRAY = 'F7F7F7';
const BOX_GRAY = 'F9F9F9';
const PEACH = 'FCE9D8';
const DEPLOY_PEACH = 'FCE9D8';
const ORANGE = 'EF771D';
const RED = 'B64231';

function lineOpts(color = BLACK, width = 1.25, extra = {}) {
  return { color, width, ...extra };
}

function addRect(x, y, w, h, fill, lineColor = BLACK, width = 1.25, rounded = true, extra = {}) {
  slide.addShape(rounded ? ST.roundRect : ST.rect, {
    ...px(x, y, w, h),
    fill: { color: fill },
    line: lineOpts(lineColor, width),
    ...extra,
  });
}

function addText(text, x, y, w, h, opts = {}) {
  slide.addText(text, {
    ...px(x, y, w, h),
    margin: 0,
    fontFace: FONT,
    fontSize: 14,
    color: BLACK,
    align: 'center',
    valign: 'mid',
    fit: 'shrink',
    breakLine: false,
    ...opts,
  });
}

function run(text, opts = {}) {
  return { text, options: { ...opts } };
}

function sub(text) {
  return run(text, { subscript: true, fontSize: 9 });
}

function sup(text) {
  return run(text, { superscript: true, fontSize: 9 });
}

function mathRuns(runs, x, y, w, h, opts = {}) {
  slide.addText(runs, {
    ...px(x, y, w, h),
    margin: 0,
    fontFace: MATH,
    fontSize: 14,
    color: BLACK,
    align: 'center',
    valign: 'mid',
    fit: 'shrink',
    ...opts,
  });
}

function addArrow(x1, y1, x2, y2, color = BLACK, width = 1.45, dashType = 'solid') {
  slide.addShape(ST.line, {
    ...px(x1, y1, x2 - x1, y2 - y1),
    line: lineOpts(color, width, { dashType, endArrowType: 'triangle' }),
  });
}

function addPlainLine(x1, y1, x2, y2, color = BLACK, width = 1.2, dashType = 'solid') {
  slide.addShape(ST.line, {
    ...px(x1, y1, x2 - x1, y2 - y1),
    line: lineOpts(color, width, { dashType }),
  });
}

function boxText(text, x, y, w, h, opts = {}) {
  addText(text, x, y, w, h, opts);
}

// Stage separators and section bars.
addPlainLine(587, 246, 587, 988, 'C7C7C7', 1.0, 'dash');
addPlainLine(1169, 246, 1169, 1090, 'C7C7C7', 1.0, 'dash');

addRect(75, 141, 478, 58, PALE_YELLOW, PALE_YELLOW, 0, false);
addRect(628, 141, 533, 58, PALE_YELLOW, PALE_YELLOW, 0, false);
addRect(1235, 141, 727, 58, PALE_YELLOW, PALE_YELLOW, 0, false);

addText('SITA: relevance-weighted source initialization with target-only inference', 250, 58, 1548, 54, {
  fontFace: FONT,
  fontSize: 24,
  bold: true,
  color: BLACK,
});
addText('Stage 1 | relevant source retrieval', 75, 145, 478, 48, {
  fontFace: UI,
  fontSize: 17,
  color: BLACK,
});
addText('Stage 2 | source-informed initialization', 628, 145, 533, 48, {
  fontFace: UI,
  fontSize: 17,
  color: BLACK,
});
addText('Stage 3 | target adaptation and target-only inference', 1235, 145, 727, 48, {
  fontFace: UI,
  fontSize: 17,
  color: BLACK,
});

// Stage 1: retrieval.
addRect(28, 255, 251, 103, PALE_GREEN, GREEN, 1.15);
addRect(290, 255, 286, 103, PALE_BLUE, BLUE, 1.15);
boxText('target — new product', 38, 270, 231, 28, { fontSize: 14, bold: true });
boxText('(limited early observations)', 37, 305, 233, 27, { fontSize: 11.5 });
boxText('source — mature products', 300, 270, 266, 28, { fontSize: 14, bold: true });
boxText('(sufficient historical observations)', 298, 305, 270, 27, { fontSize: 11.5 });

// Join arrows into aligned windows.
addPlainLine(153, 358, 153, 397);
addPlainLine(153, 397, 433, 397);
addArrow(433, 358, 433, 397);
addArrow(293, 397, 293, 439);

addRect(66, 440, 434, 154, BOX_GRAY, BLACK, 1.2);
boxText('aligned matching windows', 78, 452, 410, 26, { fontSize: 14, bold: true });
mathRuns([
  run('U', { italic: true }), sub('i'), sup('tar'), run(',  '),
  run('U', { italic: true }), sub('i'), sup('src'),
], 93, 482, 380, 31, { fontSize: 14 });
boxText('task-local MinMax scaling + flattening', 83, 518, 400, 25, { fontSize: 12.5 });
mathRuns([
  run('u', { italic: true }), sub('i'), sup('tar'), run(',  '),
  run('u', { italic: true }), sub('i'), sup('src'),
], 93, 548, 380, 29, { fontSize: 14 });

addArrow(283, 594, 283, 636);
addRect(72, 637, 424, 138, BOX_GRAY, BLACK, 1.2);
boxText('distance ranking', 84, 649, 400, 27, { fontSize: 14, bold: true });
mathRuns([
  run('Δ', { italic: true }), sub('i'), run(' = ‖', { italic: true }),
  run('u', { italic: true }), sub('i'), sup('tar'), run(' − ', { italic: true }),
  run('u', { italic: true }), sub('i'), sup('src'), run('‖', { italic: true }), sub('2'),
  run('      (5)'),
], 83, 683, 402, 31, { fontSize: 13.2 });
mathRuns([
  run('K′', { italic: true }), run(' = min(', { italic: true }), run('K', { italic: true }),
  run(', '), run('N', { italic: true }), sub('src'), run(')      (6)'),
], 83, 719, 402, 29, { fontSize: 13.2 });

addArrow(284, 775, 284, 835);
addRect(107, 835, 353, 101, PALE_BLUE, BLUE, 1.15);
boxText('selected source indices', 119, 847, 329, 27, { fontSize: 14, bold: true });
mathRuns([
  run('𝒥', { italic: true }), run(' = {', { italic: true }), run('i', { italic: true }),
  run('(1), …, '), run('i', { italic: true }), run('('), run('K′', { italic: true }), run(')}      (7)'),
], 119, 879, 329, 30, { fontSize: 12.7 });

// Inter-stage selection annotations and arrows.
addPlainLine(460, 864, 602, 864);
addPlainLine(602, 864, 602, 330);
addArrow(602, 330, 651, 330);
addText('selected source\nproducts', 480, 790, 107, 65, {
  fontFace: FONT,
  fontSize: 11.5,
  bold: true,
  color: '2674B5',
  valign: 'mid',
});
addPlainLine(496, 706, 620, 706, ORANGE, 1.45);
addPlainLine(620, 706, 620, 697, ORANGE, 1.45);
addArrow(620, 697, 659, 697, ORANGE, 1.45);
addText('corresponding\ndistances', 488, 906, 120, 57, {
  fontFace: FONT,
  fontSize: 11.5,
  color: ORANGE,
  valign: 'mid',
});

// Stage 2: source models and fusion.
addRect(651, 256, 470, 168, 'FBFBFB', BLACK, 1.2);
boxText('Source models trained on selected products', 669, 270, 434, 28, { fontSize: 13.2, bold: true });

addRect(669, 306, 125, 94, PALE_BLUE, BLUE, 1.15);
addRect(804, 306, 121, 94, PALE_BLUE, BLUE, 1.15);
addRect(976, 306, 125, 94, PALE_BLUE, BLUE, 1.15);
boxText('SRC 1', 679, 317, 105, 20, { fontSize: 12.5, bold: true });
boxText('1D-CNN', 679, 341, 105, 20, { fontSize: 12.2 });
boxText('(MSE)', 679, 365, 105, 20, { fontSize: 11.3 });
boxText('SRC 2', 813, 317, 103, 20, { fontSize: 12.5, bold: true });
boxText('1D-CNN', 813, 341, 103, 20, { fontSize: 12.2 });
boxText('(MSE)', 813, 365, 103, 20, { fontSize: 11.3 });
boxText('SRC K', 986, 317, 105, 20, { fontSize: 12.5, bold: true });
boxText('1D-CNN', 986, 341, 105, 20, { fontSize: 12.2 });
boxText('(MSE)', 986, 365, 105, 20, { fontSize: 11.3 });
addText('⋯', 932, 334, 35, 33, { fontFace: MATH, fontSize: 21, bold: true });

addArrow(731, 400, 731, 448);
addArrow(865, 400, 865, 448);
addArrow(1039, 400, 1039, 448);
addRect(655, 449, 461, 100, BOX_GRAY, BLACK, 1.2);
boxText('transferable convolution parameters', 668, 464, 435, 25, { fontSize: 13.2, bold: true });
mathRuns([
  run('{', { italic: true }), run('W', { italic: true }), sub('iℓ'), run(', '),
  run('b', { italic: true }), sub('iℓ'), run('}', { italic: true }), sub('ℓ=1'), sup('ntr'),
  run('      (8)'),
], 676, 498, 420, 29, { fontSize: 13.4 });

addArrow(885, 549, 885, 586);
addRect(659, 586, 458, 208, BOX_GRAY, BLACK, 1.2);
boxText('Inverse-distance weighting', 676, 600, 424, 25, { fontSize: 13.2, bold: true });
boxText('and parameter fusion', 676, 626, 424, 25, { fontSize: 13.2, bold: true });
mathRuns([
  run('α', { italic: true }), sub('i'), run(' = (Δ', { italic: true }), sub('i'),
  run(' + ε)', { italic: true }), sup('−1'), run(' / Σ', { italic: true }), sub('j'),
  run(' (Δ', { italic: true }), sub('j'), run(' + ε)', { italic: true }), sup('−1'),
  run('      (9)'),
], 671, 663, 432, 31, { fontSize: 12.3 });
mathRuns([
  run('W', { italic: true }), sub('ℓ'), run(' = Σ', { italic: true }), sub('i'),
  run(' α', { italic: true }), sub('i'), run('W', { italic: true }), sub('iℓ'),
], 680, 704, 415, 28, { fontSize: 13.5 });
mathRuns([
  run('b', { italic: true }), sub('ℓ'), run(' = Σ', { italic: true }), sub('i'),
  run(' α', { italic: true }), sub('i'), run('b', { italic: true }), sub('iℓ'),
  run('      (10)'),
], 680, 740, 415, 28, { fontSize: 13.5 });

addArrow(888, 794, 888, 835);
addRect(658, 835, 461, 95, BOX_GRAY, BLACK, 1.2);
mathRuns([
  run('load into target encoder: ', { fontFace: FONT }),
  run('θ', { italic: true }), sub('E'), sup('(0)'),
], 670, 847, 436, 25, { fontSize: 12.8 });
boxText('(other target parameters: random initialization)      (11)', 670, 878, 436, 26, { fontSize: 11.2 });

// Route the fused initialization into target adaptation.
addPlainLine(1119, 866, 1142, 866);
addPlainLine(1142, 866, 1142, 513);
addArrow(1142, 513, 1190, 513);

// Stage 3: target adaptation.
addRect(1180, 251, 848, 483, PEACH, BLACK, 1.25);
boxText('3.1 Target adaptation (target data only)', 1230, 264, 748, 30, { fontSize: 15, bold: true });
addRect(1349, 303, 426, 56, 'FFFFFF', BLACK, 1.1);
boxText('target observations only', 1360, 313, 404, 34, { fontSize: 13.2 });
addArrow(1562, 359, 1562, 382);
addRect(1352, 382, 424, 56, 'FFFFFF', BLACK, 1.1);
mathRuns([run('input ', { fontFace: FONT }), run('X', { italic: true }), sub('t'), sup('raw')], 1362, 393, 404, 34, { fontSize: 13.3 });

// Adaptation data path.
addArrow(1480, 438, 1480, 471);
const adaptBoxes = [
  [1190, 473, 101, 78], [1308, 473, 98, 78], [1418, 472, 119, 82],
  [1549, 472, 103, 82], [1664, 472, 98, 82], [1776, 473, 107, 78], [1908, 473, 104, 78],
];
addRect(1190, 473, 101, 78, 'FFFFFF', BLACK, 1.1);
addRect(1308, 473, 98, 78, 'FFFFFF', BLACK, 1.1);
addRect(1418, 472, 119, 82, PALE_GREEN, GREEN, 1.2);
addRect(1549, 472, 103, 82, PALE_GREEN, GREEN, 1.2);
addRect(1664, 472, 98, 82, 'FFFFFF', BLACK, 1.1);
addRect(1776, 473, 107, 78, 'FFFFFF', BLACK, 1.1);
addRect(1908, 473, 104, 78, 'FFFFFF', BLACK, 1.1);
boxText('MinMax\n(target)', 1197, 486, 87, 52, { fontSize: 11.5 });
boxText('RevIN', 1315, 484, 84, 24, { fontSize: 12.5 });
mathRuns([run('(', { fontFace: FONT }), run('γ', { italic: true }), run(', '), run('β', { italic: true }), run(')')], 1315, 509, 84, 25, { fontSize: 11.2 });
boxText('Encoder', 1426, 480, 103, 25, { fontSize: 12.2 });
mathRuns([run('(', { fontFace: FONT }), run('θ', { italic: true }), sub('E'), sup('(0)'), run(')')], 1426, 510, 103, 27, { fontSize: 10.9 });
boxText('Pred(', 1558, 486, 85, 25, { fontSize: 12.2 });
mathRuns([run('ϕ', { italic: true }), run(')')], 1558, 511, 85, 24, { fontSize: 12.2 });
mathRuns([run('RevIN', { fontFace: FONT }), sup('−1')], 1672, 498, 82, 30, { fontSize: 12.3 });
mathRuns([run('ŷ', { italic: true }), sub('t+h'), sup('sc')], 1784, 495, 91, 32, { fontSize: 13 });
mathRuns([run('ŷ', { italic: true }), sub('t+h'), sup('sc')], 1916, 495, 88, 32, { fontSize: 13 });
addArrow(1291, 512, 1308, 512);
addArrow(1406, 512, 1418, 512);
addArrow(1537, 512, 1549, 512);
addArrow(1652, 512, 1664, 512);
addArrow(1762, 512, 1776, 512);
addArrow(1883, 512, 1908, 512);

addRect(1837, 592, 123, 62, 'FFFFFF', BLACK, 1.1);
boxText('Huber', 1845, 606, 107, 32, { fontSize: 12.2 });
addArrow(1829, 551, 1829, 592);
addArrow(1960, 551, 1960, 592);

addRect(1384, 643, 307, 70, PALE_GREEN, GREEN, 1.2);
boxText('parameter update', 1394, 652, 287, 24, { fontSize: 12.3 });
boxText('(target-side trainable parameters only)', 1394, 678, 287, 24, { fontSize: 10.7 });
// Green dashed optimization feedback.
addPlainLine(1899, 654, 1899, 679, GREEN, 1.35, 'dash');
addPlainLine(1899, 679, 1691, 679, GREEN, 1.35, 'dash');
addArrow(1691, 679, 1691, 679, GREEN, 1.35, 'dash');
addArrow(1604, 643, 1604, 554, GREEN, 1.35, 'dash');
addArrow(1478, 643, 1478, 554, GREEN, 1.35, 'dash');
addArrow(1360, 643, 1360, 551, GREEN, 1.35, 'dash');
addArrow(1899, 679, 1691, 679, GREEN, 1.35, 'dash');

// Adapted parameters and deployment.
addArrow(1604, 734, 1604, 773);
addRect(1414, 775, 334, 81, 'FFFFFF', BLACK, 1.15);
boxText('adapted target parameters', 1424, 786, 314, 25, { fontSize: 12.8 });
mathRuns([
  run('θ', { italic: true }), sub('E'), sup('*'), run(', ϕ', { italic: true }), sup('*'),
  run(', γ', { italic: true }), sup('*'), run(', β', { italic: true }), sup('*'),
], 1424, 817, 314, 27, { fontSize: 13 });
addArrow(1581, 856, 1581, 885);

addRect(1176, 886, 866, 176, DEPLOY_PEACH, BLACK, 1.25);
addText('3.2 deployment: target input only; no source products, source models, source selection, or parameter fusion.', 1194, 901, 830, 31, {
  fontFace: FONT,
  fontSize: 11.1,
  bold: true,
  color: RED,
  align: 'left',
  valign: 'mid',
});

// Deployment path.
addRect(1188, 955, 54, 77, 'FFFFFF', BLACK, 1.0);
addRect(1255, 955, 72, 77, 'FFFFFF', BLACK, 1.0);
addRect(1338, 955, 69, 77, 'FFFFFF', BLACK, 1.0);
addRect(1413, 946, 77, 89, PALE_GREEN, GREEN, 1.1);
addRect(1501, 946, 80, 89, PALE_GREEN, GREEN, 1.1);
addRect(1594, 955, 71, 77, 'FFFFFF', BLACK, 1.0);
addRect(1678, 955, 76, 77, 'FFFFFF', BLACK, 1.0);
addRect(1765, 955, 102, 77, 'FFFFFF', BLACK, 1.0);
addRect(1878, 955, 142, 77, 'FFFFFF', BLACK, 1.0);

mathRuns([run('input', { fontFace: FONT }), run(' '), run('X', { italic: true }), sub('t'), sup('raw')], 1191, 975, 48, 38, { fontSize: 9.2 });
boxText('MinMax\n(target)', 1260, 968, 62, 50, { fontSize: 8.6 });
mathRuns([run('RevIN', { fontFace: FONT })], 1343, 969, 59, 23, { fontSize: 9.5 });
mathRuns([run('(', { fontFace: FONT }), run('γ', { italic: true }), sup('*'), run(', '), run('β', { italic: true }), sup('*'), run(')')], 1341, 990, 63, 28, { fontSize: 8.2 });
boxText('Encoder', 1418, 959, 67, 25, { fontSize: 9.0 });
mathRuns([run('(', { fontFace: FONT }), run('θ', { italic: true }), sub('E'), sup('*'), run(')')], 1418, 985, 67, 27, { fontSize: 8.4 });
boxText('Pred(', 1507, 960, 68, 23, { fontSize: 9.0 });
mathRuns([run('ϕ', { italic: true }), sup('*'), run(')')], 1507, 985, 68, 27, { fontSize: 8.9 });
mathRuns([run('RevIN', { fontFace: FONT }), sup('−1')], 1600, 977, 59, 32, { fontSize: 9.4 });
mathRuns([run('ŷ', { italic: true }), sub('t+h'), sup('sc')], 1685, 978, 62, 30, { fontSize: 9.7 });
mathRuns([run('g', { italic: true }), sub('pred'), sup('−1')], 1770, 967, 92, 33, { fontSize: 9.3 });
boxText('(inverse scaling)', 1770, 994, 92, 24, { fontSize: 7.8 });
boxText('output prediction', 1885, 969, 128, 26, { fontSize: 9.3 });
mathRuns([run('ŷ', { italic: true }), sub('t+h'), sup('raw')], 1885, 995, 128, 27, { fontSize: 10.2 });

addArrow(1242, 993, 1255, 993, BLACK, 1.1);
addArrow(1327, 993, 1338, 993, BLACK, 1.1);
addArrow(1407, 993, 1413, 993, BLACK, 1.1);
addArrow(1490, 990, 1501, 990, BLACK, 1.1);
addArrow(1581, 993, 1594, 993, BLACK, 1.1);
addArrow(1665, 993, 1678, 993, BLACK, 1.1);
addArrow(1754, 993, 1765, 993, BLACK, 1.1);
addArrow(1867, 993, 1878, 993, BLACK, 1.1);

// Use a compact metadata note so the source and reconstruction intent remain inspectable in PowerPoint.
slide.addNotes('Editable reconstruction based on SITA_editable_flowchart(2)_slide2.png. Text, formulas, flow boxes, and connectors are separate PowerPoint objects.');

const output = '/Users/ming/Desktop/复现实验/保留的复现实验修改rfe/SITA_editable_flowchart_editable.pptx';
pptx.writeFile({ fileName: output });
