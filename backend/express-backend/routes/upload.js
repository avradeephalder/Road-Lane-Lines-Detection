const express = require('express');
const router = express.Router();
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        const dir = './uploads';
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }
        cb(null, dir);
    },
    filename: function (req, file, cb) {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, uniqueSuffix + '-' + file.originalname);
    }
});

const upload = multer({
    storage: storage,
    limits: { fileSize: 500 * 1024 * 1024 }, // 500MB
    fileFilter: (req, file, cb) => {
        const allowedTypes = /jpeg|jpg|png|mp4|avi|mov/;
        const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
        const mimetype = allowedTypes.test(file.mimetype);
        
        if (mimetype && extname) {
            return cb(null, true);
        } else {
            cb(new Error('Invalid file type. Only images and videos allowed.'));
        }
    }
});

// Flask ML service URL
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:5000';

// Upload and process endpoint
router.post('/process', upload.single('file'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
    }

    const uploadedFilePath = req.file.path;
    const outputDir = './outputs';
    
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }

    const outputFileName = `processed_${Date.now()}_${req.file.originalname}`;
    const outputFilePath = path.join(outputDir, outputFileName);

    try {
        // Create form data to send to Flask
        const formData = new FormData();
        formData.append('file', fs.createReadStream(uploadedFilePath), {
            filename: req.file.originalname,
            contentType: req.file.mimetype
        });

        // Send to Flask ML service
        const response = await axios.post(`${ML_SERVICE_URL}/predict`, formData, {
            headers: {
                ...formData.getHeaders()
            },
            responseType: 'stream',
            maxContentLength: Infinity,
            maxBodyLength: Infinity,
            timeout: 600000 // 10 minutes timeout for video processing
        });

        // Save processed file
        const writer = fs.createWriteStream(outputFilePath);
        response.data.pipe(writer);

        await new Promise((resolve, reject) => {
            writer.on('finish', resolve);
            writer.on('error', reject);
        });

        // Clean up uploaded file
        fs.unlinkSync(uploadedFilePath);

        // Send processed file back to frontend
        res.download(outputFilePath, outputFileName, (err) => {
            if (err) {
                console.error('Download error:', err);
            }
            // Clean up after sending
            setTimeout(() => {
                if (fs.existsSync(outputFilePath)) {
                    fs.unlinkSync(outputFilePath);
                }
            }, 5000);
        });

    } catch (error) {
        console.error('Processing error:', error);
        
        // Clean up on error
        if (fs.existsSync(uploadedFilePath)) {
            fs.unlinkSync(uploadedFilePath);
        }
        if (fs.existsSync(outputFilePath)) {
            fs.unlinkSync(outputFilePath);
        }

        res.status(500).json({
            error: 'Processing failed',
            details: error.message
        });
    }
});

// Check ML service health
router.get('/ml-health', async (req, res) => {
    try {
        const response = await axios.get(`${ML_SERVICE_URL}/health`);
        res.json(response.data);
    } catch (error) {
        res.status(503).json({ error: 'ML service unavailable' });
    }
});

module.exports = router;
