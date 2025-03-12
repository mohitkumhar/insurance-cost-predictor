# 🚀 Insurance Cost Predictor  

**Insurance Cost Predictor** is a machine learning-based web application that estimates health insurance costs using TensorFlow and Streamlit. It predicts the cost based on user inputs like **age, sex, BMI, number of children, smoking status**, and **region**.  

## 📌 **How It Works**  
1. User inputs are encoded using **OneHotEncoding** (for sex and smoking status) and **LabelEncoding** (for region).  
2. Data is scaled using **scikit-learn** scalers.  
3. The pre-trained TensorFlow model predicts the insurance cost.  
4. Prediction is scaled back to the original range and displayed to the user.  

## 🛠️ **Technologies Used**  
- TensorFlow  
- Streamlit  
- Scikit-learn  
- Pandas  
- NumPy  

## 🎯 **Key Features**  
✅ Real-time prediction  
✅ User-friendly interface  
✅ Handles encoding, scaling, and prediction seamlessly  
✅ Fast and accurate results  

## 🚀 **How to Run**  
1. Clone the repository:  
```bash
git clone https://github.com/mohitkumhar/insurance-cost-predictor.git
```
2. Install dependencies:  
```bash
pip install -r requirements.txt
```
3. Run the app:  
```bash
streamlit run app.py
```

## 📄 **Future Improvements**  
- Add more user-friendly UI enhancements  
- Improve model accuracy with more training data  
- Add graphical representation of predictions  

---

Contributions and feedback are welcome! 😎
