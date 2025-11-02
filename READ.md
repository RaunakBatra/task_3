# 🍽️ Food Recommendation System - Backend

This backend powers the **Food Recommendation System**, providing authentication, user management, email verification, password reset, and recipe data APIs.  
Built with **Node.js**, **Express.js**, **MongoDB**, and **Nodemailer**.

---

## 🚀 Features

✅ User Registration, Login & Logout  
✅ Email Verification using OTP  
✅ Resend OTP functionality  
✅ Password Reset using OTP  
✅ Fetch User Details  
✅ Fetch Recipes for Recommendation  
✅ MongoDB Integration  
✅ Ready for ML API integration  

---

## 🗂️ Folder Structure

```
task-4/
└── server/
    ├── index.js
    ├── .env
    ├── package.json
    ├── config/
    │   ├── connection.js
    │   └── nodemailer.js
    ├── controllers/
    │   ├── authController.js
    │   ├── recipeController.js
    │   └── userController.js
    ├── middleware/
    │   └── usermiddle.js
    ├── models/
    │   ├── usermodels.js
    │   └── recipeModel.js
    ├── routes/
    │   ├── authRoutes.js
    │   ├── userRoutes.js
    │   └── recipeRoutes.js
    ├── scripts/
    │   └── importData.js
    └── recipe_final_processed.json
```

---

## ⚙️ Setup Instructions

### 1️⃣ Install dependencies
```bash
cd server
npm install
```

### 2️⃣ Create `.env` file
```env
PORT=8000
MONGODB_URI="mongodb://127.0.0.1:27017/task4"
JWT_SECRET="secret#text"

SMTP_USER="your-brevo-user"
SMTP_PASS="your-brevo-password"
SENDER_EMAIL="your-email@gmail.com"

ML_API_BASE="http://127.0.0.1:8000"
```

### 3️⃣ Run server
```bash
npm run dev
```
Server runs on: `http://localhost:8000`

---

## 🔑 Authentication APIs

### 🧩 Register User
**Endpoint:** `POST /api/auth/register`

**Request:**
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "password123"
}
```

**Response:**
```json
{
  "success": true,
  "message": "User registered successfully"
}
```

**Frontend (Axios):**
```js
await axios.post('/api/auth/register', { name, email, password }, { withCredentials: true });
```

---

### 🔑 Login User
**Endpoint:** `POST /api/auth/login`

**Request:**
```json
{
  "email": "john@example.com",
  "password": "password123"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Login successful"
}
```

**Frontend:**
```js
await axios.post('/api/auth/login', { email, password }, { withCredentials: true });
```

---

### 🚪 Logout User
**Endpoint:** `POST /api/auth/logout`

**Response:**
```json
{ "success": true, "message": "Logged out successfully" }
```

**Frontend:**
```js
await axios.post('/api/auth/logout', {}, { withCredentials: true });
```

---

### ✉️ Send Verification OTP
**Endpoint:** `POST /api/auth/sendverifyotp`  
🔒 Requires Auth Token (cookie)

**Response:**
```json
{
  "success": true,
  "message": "Verification OTP sent to your email"
}
```

---

### ✅ Verify Account
**Endpoint:** `POST /api/auth/verifyaccount`  
🔒 Requires Auth Token

**Request:**
```json
{ "otp": "123456" }
```

**Response:**
```json
{ "success": true, "message": "Email verified successfully" }
```

---

### 🔁 Resend Verification OTP
**Endpoint:** `POST /api/auth/resendverifyotp`  
🔒 Requires Auth Token

**Response:**
```json
{ "success": true, "message": "New OTP sent successfully" }
```

---

### 🔒 Send Password Reset OTP
**Endpoint:** `POST /api/auth/sendResetOtp`

**Request:**
```json
{ "email": "john@example.com" }
```

**Response:**
```json
{ "success": true, "message": "OTP sent to your email for password reset" }
```

---

### 🔑 Reset Password
**Endpoint:** `POST /api/auth/resetPassword`

**Request:**
```json
{
  "email": "john@example.com",
  "otp": "123456",
  "password": "newPassword123"
}
```

**Response:**
```json
{ "success": true, "message": "Password has been reset successfully" }
```

---

## 👤 User Routes

### 🔍 Get User Data
**Endpoint:** `GET /api/user/data`  
🔒 Requires Auth Token

**Response:**
```json
{
  "success": true,
  "userData": {
    "name": "John Doe",
    "isAccountVerified": true
  }
}
```

**Frontend:**
```js
const res = await axios.get('/api/user/data', { withCredentials: true });
console.log(res.data.userData);
```

---

## 🍳 Recipe Routes

### 📦 Get All Recipes
**Endpoint:** `GET /api/recipes`

**Response:**
```json
[
  {
    "recipe_id": 101,
    "recipe_name": "Pasta Alfredo",
    "aver_rate": 4.8,
    "image_url": "https://images.example.com/pasta.jpg",
    "ingredients_list": "['pasta', 'cheese', 'cream']",
    "veg_nonveg": "Veg",
    "cuisine_type": "Italian",
    "region_type": "Europe"
  },
  {
    "recipe_id": 102,
    "recipe_name": "Chicken Biryani",
    "aver_rate": 4.9,
    "image_url": "https://images.example.com/biryani.jpg",
    "veg_nonveg": "Non-Veg",
    "cuisine_type": "Indian",
    "region_type": "Asia"
  }
]
```

**Frontend Example:**
```js
const res = await axios.get('/api/recipes');
setRecipes(res.data); // Store recipes in state
```

---

## 🧾 Example `.env` File

```env
MONGODB_URI="mongodb://127.0.0.1:27017/task4"
JWT_SECRET="secret#text"
SMTP_USER="9a5639001@smtp-brevo.com"
SMTP_PASS="your-password"
SENDER_EMAIL="chauhanujjawal070@gmail.com"
PORT=8000
ML_API_BASE="http://127.0.0.1:8000"
```

---

## ⚠️ Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Cannot GET /api/auth/register` | Sent GET request instead of POST | Use POST method |
| `User not found` | Invalid email or unregistered account | Register first |
| `Invalid OTP` | Wrong or expired OTP | Request new OTP |
| `CORS Error` | Missing `credentials: true` in frontend | Add it to Axios config |

---

## 🧠 Frontend Integration Notes

- Always include `{ withCredentials: true }` for requests that require login.  
- JWT is stored as an **HTTP-only cookie** — cannot be accessed from JavaScript.  
- Handle `401` or `403` responses by redirecting to the login page.  
- Recipe data can be displayed in cards or filtered using cuisine type, calories, etc.

---

## 👨‍💻 Developer
**Ujjawal Chauhan**  
Backend Developer | Food Recommendation System

---
