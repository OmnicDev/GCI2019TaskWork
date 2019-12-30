const express = require("express");
const cors = require("cors");
const path = require("path");

const app = express();

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(cors());
app.use("/", require("./routes/main.router"));
app.use(express.static("public"));

const PORT = process.env.PORT || 8080;

app.listen(PORT, () => {
    console.log(`Server running on Port ${PORT}`)
})