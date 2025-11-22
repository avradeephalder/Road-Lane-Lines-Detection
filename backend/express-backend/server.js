const express = require('express');
const cors = require('cors');
const uploadRouter = require('./routes/upload');

const app = express();
const PORT = process.env.PORT || 4000;

// Middleware
app.use(cors({
    origin: 'http://localhost:3000',
    credentials: true
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.use('/api/lane-detection', uploadRouter);

// Health check
app.get('/health', (req, res) => {
    res.json({ status: 'ok', message: 'Express backend running' });
});

app.listen(PORT, () => {
    console.log(`Express server running on port ${PORT}`);
});
