# 🍎 AI Food Calorie Estimator - Food-101

> _"Because counting calories by hand is so 1995."_

---

## 📸 App Screenshots

Sometimes seeing is believing. Here are some glimpses of the app in action:

<p align="center">
  <img src="assests\Home-Page.png" width="45%" alt="Screenshot 1" />
  <img src="assests\Training.png" width="45%" alt="Screenshot 2" /><br>
  <img src="assests\Training2.png" width="45%" alt="Screenshot 3" />
  <img src="assests\Meal%20Tracker.png" width="45%" alt="Screenshot 4" /><br>
  <img src="assests\About.png" width="45%" alt="Screenshot 5" />
</p>

*Each frame brought to you by AI magic and way too much coffee.*

## 🚀 Demo Video

🎬 **Watch the magic happen!**  
[![Demo Video](https://img.youtube.com/vi/your-demo-video-id/0.jpg)](https://www.youtube.com/watch?v=your-demo-video-id)  
*Click above to see the app in action. Yes, it really works. Sometimes even on pizza.*<br>
**The Video will be uploaded soon**

---

## 🏗️ Project Structure

Ever wondered what a well-organized AI project looks like? Well, keep wondering. But here’s what you get:

```
.
├── app.py                  # The Streamlit app (the main event)
├── config.yaml             # All your settings, because hardcoding is for amateurs
├── dataset_utils.py        # Data wrangling, validation, and other wizardry
├── food101_downloader.py   # Downloads and organizes Food-101 (so you don’t have to)
├── model_trainer.py        # Model training, evaluation, and existential crises
├── requirements.txt        # All the packages you’ll forget to install
├── setup.py                # Project setup and environment checks
├── training_config.py      # Training presets and hyperparameter drama
├── data/
│   └── food101_calories.csv # Nutrition facts (so you can feel guilty)
├── downloads/
│   └── food-101.tar.gz     # The big, beautiful dataset
├── food-101/
│   ├── images/             # 101,000+ food pics (don’t look hungry)
│   └── meta/               # Official splits and class lists
├── organized_food101/
│   ├── train/
│   ├── val/
│   └── test/
├── models/                 # Where your trained models live (if you ever finish training)
├ logs/                   # For when things go wrong (they will)
└── .vscode/                # VSCode settings (because we all use it anyway)
```

---

### Note:
**This Project is still under refinement and optimization**

## 🏃 How to Run (a.k.a. “Just Make It Work”)

1. **Clone this repo**  
   ```
   git clone https://github.com/DevanshSrajput/ai-food-calorie-estimator.git
   cd ai-food-calorie-estimator
   ```

2. **Install dependencies**  
   _Pro tip: Use a virtual environment. Or don’t, and enjoy dependency hell._
   ```
   pip install -r requirements.txt
   ```

3. **Setup the project**  
   _Let the script do the heavy lifting (and the complaining)._
   ```
   python setup.py
   ```

4. **Download and organize the Food-101 dataset**  
   _This will take a while. Go make a sandwich._
   ```
   python food101_downloader.py --all
   ```

5. **Run the app**  
   _The moment of truth (or disappointment)._
   ```
   streamlit run app.py
   ```

6. **Open your browser**  
   Visit [http://localhost:8501](http://localhost:8501)  
   _If it doesn’t open automatically, you know what to do._

---

## 🎉 Features

- **Food Recognition:** 101 food categories. Yes, even “hot_dog”.
- **Custom Model Training:** Because the default isn’t good enough for you.
- **Nutrition Tracking:** Log your meals and pretend you’ll eat healthier tomorrow.
- **Analytics:** Pie charts, bar charts, and other ways to visualize your guilt.
- **Export & Backup:** For when you want to share your calorie shame with others.
- **Streamlit UI:** Looks good, works fast, and occasionally crashes (just kidding… mostly).

---

## 🤖 Tech Stack

- **TensorFlow 2.13+** – Deep learning, but deeper.
- **Streamlit** – For beautiful UIs with minimal effort.
- **OpenCV, Pillow** – Image wrangling.
- **Pandas, NumPy** – Data science essentials.
- **Plotly, Matplotlib, Seaborn** – Because one plotting library is never enough.

---

## 🧩 Training Presets

Pick your poison in `training_config.py` or the UI:

- **Quick:** For the impatient.
- **Balanced:** For the indecisive.
- **High Accuracy:** For those with time (and a GPU farm).
- **Lightweight:** For Raspberry Pi dreamers.
- **Production:** For the 1% who’ll actually deploy this.
- **Research:** For PhDs and masochists.

---

## 🥗 Nutrition Database

- Calories, protein, carbs, fats for every class.
- Data in `data/food101_calories.csv`.
- Now you can know exactly how much you regret that “cheesecake”.

---

## 🛠️ Development & Customization

- **Config everything** in `config.yaml`.  
- **Add new models** in `model_trainer.py` (if you’re brave).
- **Balance classes, validate data, and more** with `dataset_utils.py`.

---

## 🐞 Troubleshooting

- **TensorFlow won’t install?**  
  Try upgrading pip, or just cry a little.
- **Out of memory?**  
  Welcome to deep learning.
- **App won’t start?**  
  Check your Python version. Or your karma.

---

## 📢 Credits

- **DevanshSrajput** – Project lead, code wrangler, and chief sarcasm officer.
- **ETH Zurich** – For the [Food-101 dataset](https://www.vision.ee.ethz.ch/datasets_extra/food-101/).

---

## 📜 License

MIT. Use it, break it, fix it, share it. Just don’t blame us if you gain weight.

---

## 💡 Final Thoughts

If you made it this far, you’re either really interested or really lost. Either way, enjoy the project, and remember:  
**Calories don’t count if you don’t log**