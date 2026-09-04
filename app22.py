import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import joblib
import json
import os
from PIL import Image

# =========================================================================
# 0. 基本頁面設定（合併原本重複呼叫的 set_page_config，避免 Streamlit 報錯）
# =========================================================================
icon = Image.open("favicon.ico")

pgtitle = "目標:心理壓力"
pgmenu1 = "1. 研究"
pgmenu2 = "2. 偵測"

st.set_page_config(
    page_title=pgtitle,
    page_icon=icon,
    layout="wide"
)

# =========================================================================
# 1. 載入模型與特徵清單
# =========================================================================
@st.cache_resource
def load_assets():
    model = joblib.load('lr_top8_model.pkl')
    with open('top8_features.json', 'r', encoding='utf-8') as f:
        feature_names = json.load(f)
    return model, feature_names

try:
    model, top8_cols = load_assets()
except Exception as e:
    st.error("🚨 找不到模型檔案或欄位清單！請確保先在 Jupyter 執行過存檔腳本。")
    st.stop()

# =========================================================================
# 2. 側邊欄選單（比照 app14.py 做法：標題 + 垂直選單）
# =========================================================================
st.sidebar.markdown(
    f"""
    <div style="
        font-size: 20px;
        font-weight: 700;
        white-space: nowrap;
        margin-bottom: 14px;
    ">
        🧠 第四組
    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    " ",
    [pgmenu1, pgmenu2]
)

# =========================================================================
# 3. 共用資料 / 小工具（研究頁與偵測頁都會用到）
# =========================================================================
name_mapping = {
    'bmi': '身體質量指數 (BMI)',
    'sleep_duration_hrs': '睡眠時數 (小時)',
    'sleep_quality_score': '睡眠品質分數 (0~100)',
    'alcohol_units_before_bed': '睡前飲酒量 (單位)',
    'cognitive_performance_score': '認知表現分數 (0~100)',
    'occupation_Retired': '是否退休 (Retired)',
    'occupation_Lawyer': '職業：律師 (Lawyer)',
    'occupation_Nurse': '職業：護理師 (Nurse)'
}

num_features = ['age', 'bmi', 'sleep_duration_hrs', 'sleep_quality_score', 'rem_percentage',
                'deep_sleep_percentage', 'sleep_latency_mins', 'wake_episodes_per_night',
                'caffeine_mg_before_bed', 'alcohol_units_before_bed', 'screen_time_before_bed_mins',
                'steps_that_day', 'nap_duration_mins', 'work_hours_that_day', 'heart_rate_resting_bpm',
                'room_temperature_celsius', 'weekend_sleep_diff_hrs', 'cognitive_performance_score']

continuous_cols = [c for c in top8_cols if c in num_features]
categorical_cols = [c for c in top8_cols if c not in num_features]
sorted_top8_cols = continuous_cols + categorical_cols


def rescale_to_model(val, min_val, max_val):
    if max_val == min_val:
        return 0.5
    return 0.2 + 0.6 * (val - min_val) / (max_val - min_val)


# =========================================================================
# PAGE 1：研究（文字展現）
# =========================================================================
if page == pgmenu1:

    st.title("目標:心理壓力")
    st.caption("研究說明：從大規模睡眠與生活行為資料，建立一套可解釋的壓力指數預測模型。")

    st.subheader("0. 研究背景與資料集")

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("樣本數", "100,000")
    with col_b:
        st.metric("原始特徵", "30+")
    with col_c:
        st.metric("模型類型", "線性迴歸")
    with col_d:
        st.metric("精簡後特徵", str(len(top8_cols)))

    st.markdown(
        """
        本研究以睡眠品質、生活作息與職業背景等多面向資料，訓練一套可解釋的
        **線性迴歸模型**，用來推估受測者的壓力指數（Stress Score）。原始模型涵蓋
        數十個欄位（含 One-hot 編碼後的類別欄位），為了兼顧**即時互動體驗**與
        **模型可解釋性**，本系統的即時預測頁面採用精簡後、僅保留最具影響力
        8 個特徵的簡化版模型。
        """
    )

    st.subheader("1. 模型精簡流程")
    st.markdown(
        """
        1. **特徵重要性排序**：先在完整模型上計算每個特徵對壓力指數的影響力（迴歸係數絕對值 / 重要性分數）。
        2. **挑選前 8 大關鍵特徵**：保留對預測結果影響最大的欄位，捨棄影響力極小的欄位以降低雜訊與模型複雜度。
        3. **重新訓練精簡模型**：僅用這幾個欄位重新訓練線性迴歸模型（`lr_top8_model.pkl`），
           並將對應欄位清單存成 `top8_features.json`，供即時預測頁面讀取使用。
        4. **輸入數值轉換**：連續型欄位（如睡眠時數、BMI）會先正規化，轉換到模型訓練時使用的數值區間，
           確保使用者在介面上拖動的滑桿數值，能正確對應到模型看得懂的輸入格式。
        """
    )

    st.subheader("2. 本模型使用的核心特徵")

    feat_rows = []
    for col in sorted_top8_cols:
        feat_rows.append({
            "特徵欄位": col,
            "顯示名稱": name_mapping.get(col, col),
            "類型": "連續數值" if col in num_features else "二元 / 類別（是或否）"
        })

    st.dataframe(pd.DataFrame(feat_rows), width="stretch", hide_index=True)

    st.subheader("3. 壓力等級判讀標準")
    st.markdown(
        """
        模型預測結果會換算為 0～100 分的壓力指數，並依區間分為四個等級：

        | 區間 | 等級 | 說明 |
        |---|---|---|
        | 0 – 30 | 🟢 良好（HEALTHY） | 壓力狀態穩定，無明顯風險 |
        | 30 – 55 | 🟡 輕度（MILD） | 略有壓力徵兆，建議持續留意作息 |
        | 55 – 80 | 🟠 中度（MODERATE） | 壓力偏高，建議調整生活習慣 |
        | 80 – 100 | 🔴 高度（SEVERE） | 壓力風險高，建議尋求專業協助 |
        """
    )

    st.info("👉 想實際體驗模型預測，請切換左側選單至「2. 偵測」。")

    st.caption("⚠️ 本頁內容為模型方法論之研究說明，非醫療診斷建議。")

# =========================================================================
# PAGE 2：偵測（輸入 + 結果，合併為同一頁的左右兩欄）
# =========================================================================
elif page == pgmenu2:

    st.title("⏱️ 即時壓力評估偵測")
    st.caption("調整左側欄位的健康與行為指標，右側儀表會即時反映預測出的壓力指數。")

    col_input, col_result = st.columns([1, 1.1])

    user_inputs = {}

    with col_input:
        st.markdown("#### 📊 請輸入受測者健康與行為指標")

        for col in sorted_top8_cols:
            display_name = name_mapping.get(col, col)

            # A. 連續型數值特徵
            if col in num_features:
                if col == 'cognitive_performance_score' or col == 'sleep_quality_score':
                    raw_val = st.slider(f"📈 {display_name}", min_value=0.0, max_value=100.0, value=75.0)
                    user_inputs[col] = rescale_to_model(raw_val, 0.0, 100.0)

                elif col == 'sleep_duration_hrs':
                    raw_val = st.slider(f"📈 {display_name}", min_value=0.0, max_value=12.0, value=7.0, step=0.5)
                    user_inputs[col] = rescale_to_model(raw_val, 0.0, 12.0)

                elif col == 'alcohol_units_before_bed':
                    raw_val = st.slider(f"📈 {display_name}", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
                    user_inputs[col] = rescale_to_model(raw_val, 0.0, 10.0)

                elif col == 'bmi':
                    raw_val = st.slider(f"📈 {display_name}", min_value=15.0, max_value=40.0, value=22.5, step=0.1)
                    user_inputs[col] = rescale_to_model(raw_val, 15.0, 40.0)

                elif col == 'age':
                    raw_val = st.slider(f"📈 {display_name}", min_value=18.0, max_value=100.0, value=40.0, step=1.0)
                    user_inputs[col] = rescale_to_model(raw_val, 18.0, 100.0)

                else:
                    user_inputs[col] = st.slider(f"📈 {display_name}", 0.2, 0.8, 0.5)

            # B. 二元型特徵（是或否）
            elif col in ['exercise_day', 'sleep_aid_used', 'shift_work', 'felt_rested']:
                choice = st.selectbox(f"🔄 {display_name}", options=["否 (0)", "是 (1)"])
                user_inputs[col] = 1.0 if "是" in choice else 0.0

            # C. 其餘 One-Hot 分類特徵（職業勾選等）
            else:
                choice = st.checkbox(f"📍 {display_name}")
                user_inputs[col] = 0.8 if choice else 0.2

    # ---------------------------------------------------------------
    # 後台即時 AI 預測與分數轉換邏輯
    # ---------------------------------------------------------------
    df_input = pd.DataFrame([user_inputs])[top8_cols]
    y_pred_raw = model.predict(df_input)

    if y_pred_raw <= 1.0:
        final_gauge_score = ((y_pred_raw - 0.2) / 0.6) * 100
    else:
        final_gauge_score = (y_pred_raw / 10.0) * 100

    final_gauge_score = max(0.0, min(100.0, float(final_gauge_score.item())))
    final_gauge_score = float(final_gauge_score)

    # ---------------------------------------------------------------
    # 前端客製化：動態注入 amCharts 4 程式碼
    # ---------------------------------------------------------------
    html_code = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="https://cdn.amcharts.com/lib/4/core.js"></script>
    <script src="https://cdn.amcharts.com/lib/4/charts.js"></script>
    <script src="https://cdn.amcharts.com/lib/4/themes/animated.js"></script>
    <style>
       body {{
      background-color: #0d1117;
      margin: 0;
      padding: 0;
      overflow: hidden;
    }}

    #chartdiv {{
      width: 100%;
      height: 500px;
      background: #0d1117;
      border-radius: 16px;
      border: 1px solid #30363d;
      box-shadow: 0 0 30px rgba(0, 242, 254, 0.15);
      position: relative;
    }}
    </style>
</head>
<body>

    <!-- SVG 霓虹發光濾鏡 -->
    <svg style="position: absolute; width: 0; height: 0;">
        <filter id="super-neon-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="6" result="blur" />
            <feMerge>
                <feMergeNode in="blur"/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>
        </filter>
    </svg>

    <div id="chartdiv"></div>

    <script>
    am4core.ready(function() {{
    am4core.useTheme(am4themes_animated);

    var chartMin = 0;
    var chartMax = 100;
    var realModelScore = {final_gauge_score};

    var neonColors = {{
      healthy: "#00f2fe",
      mild: "#00ff87",
      moderate: "#fdae19",
      severe: "#ff0055"
    }};

    var data = {{
      score: realModelScore,
      gradingData: [
        {{ title: "HEALTHY", color: neonColors.healthy, lowScore: 0, highScore: 30 }},
        {{ title: "MILD", color: neonColors.mild, lowScore: 30, highScore: 55 }},
        {{ title: "MODERATE", color: neonColors.moderate, lowScore: 55, highScore: 80 }},
        {{ title: "SEVERE", color: neonColors.severe, lowScore: 80, highScore: 100 }}
      ]
    }};

    function lookUpGrade(lookupScore, grades) {{
      for (var i = 0; i < grades.length; i++) {{
        if (grades[i].lowScore <= lookupScore && grades[i].highScore >= lookupScore) {{
          return grades[i];
        }}
      }}
      return grades[grades.length - 1];
    }}

    var chart = am4core.create("chartdiv", am4charts.GaugeChart);
    chart.hiddenState.properties.opacity = 0;
    chart.fontSize = 11;

    chart.padding(40, 40, 40, 40);
    chart.innerRadius = am4core.percent(85);
    chart.resizable = true;

    var axis = chart.xAxes.push(new am4charts.ValueAxis());
    axis.min = chartMin;
    axis.max = chartMax;
    axis.strictMinMax = true;
    axis.renderer.radius = am4core.percent(95);
    axis.renderer.inside = true;

    axis.renderer.line.strokeOpacity = 0.15;
    axis.renderer.line.stroke = am4core.color("#58a6ff");
    axis.renderer.line.strokeWidth = 1;

    axis.renderer.ticks.template.disabled = false;
    axis.renderer.ticks.template.strokeOpacity = 0.4;
    axis.renderer.ticks.template.stroke = am4core.color("#58a6ff");
    axis.renderer.ticks.template.length = 6;

    axis.renderer.grid.template.disabled = false;
    axis.renderer.grid.template.opacity = 0.02;
    axis.renderer.grid.template.stroke = am4core.color("#fff");

    axis.renderer.labels.template.radius = am4core.percent(12);
    axis.renderer.labels.template.fontSize = "0.85em";
    axis.renderer.labels.template.fontFamily = "monospace";
    axis.renderer.labels.template.fill = am4core.color("#8b949e");

    var axis2 = chart.xAxes.push(new am4charts.ValueAxis());
    axis2.min = chartMin;
    axis2.max = chartMax;
    axis2.strictMinMax = true;
    axis2.renderer.labels.template.disabled = true;
    axis2.renderer.ticks.template.disabled = true;
    axis2.renderer.grid.template.disabled = true;

    for (let grading of data.gradingData) {{
      var range = axis2.axisRanges.create();
      var baseColor = am4core.color(grading.color);

      range.axisFill.fill = baseColor;
      range.axisFill.fillOpacity = 0.2;
      range.axisFill.zIndex = -1;
      range.value = grading.lowScore;
      range.endValue = grading.highScore;

      range.axisFill.stroke = baseColor;
      range.axisFill.strokeWidth = 3;
      range.axisFill.strokeOpacity = 1;

      range.axisFill.dom.setAttribute("filter", "url(#super-neon-glow)");
    }}

    var matchingGrade = lookUpGrade(data.score, data.gradingData);

    var label = chart.radarContainer.createChild(am4core.Label);
    label.isMeasured = false;
    label.fontSize = "5.5em";
    label.fontFamily = "monospace";
    label.x = am4core.percent(50);
    label.paddingBottom = 25;
    label.horizontalCenter = "middle";
    label.verticalCenter = "bottom";
    label.text = data.score.toFixed(1);
    label.fill = am4core.color(matchingGrade.color);
    label.dom.setAttribute("filter", "url(#super-neon-glow)");

    var label2 = chart.radarContainer.createChild(am4core.Label);
    label2.isMeasured = false;
    label2.fontSize = "1.4em";
    label2.fontFamily = "monospace";
    label2.fontWeight = "bold";
    label2.horizontalCenter = "middle";
    label2.verticalCenter = "bottom";
    label2.text = "";
    label2.fill = am4core.color(matchingGrade.color);
    label2.dom.setAttribute("filter", "url(#super-neon-glow)");

    var hand = chart.hands.push(new am4charts.ClockHand());
    hand.axis = axis2;
    hand.innerRadius = am4core.percent(50);
    hand.startWidth = 5;
    hand.endWidth = 1;
    hand.pin.disabled = true;
    hand.fill = am4core.color("#ffffff");
    hand.stroke = am4core.color("#ffffff");
    hand.dom.setAttribute("filter", "url(#super-neon-glow)");

    setTimeout(function() {{
        hand.showValue(data.score, 2000, am4core.ease.cubicOut);
    }}, 400);

    hand.events.on("positionchanged", function(){{
      var currentVal = axis2.positionToValue(hand.currentPosition);
      label.text = currentVal.toFixed(1);
      var matchingGrade = lookUpGrade(currentVal, data.gradingData);
      var targetColor = am4core.color(matchingGrade.color);

      var chiStatus = "";
      if (currentVal <= 30) chiStatus = "🟢 SYSTEM_STATUS: 良好 (HEALTHY)";
      else if (currentVal <= 55) chiStatus = "🟡 SYSTEM_STATUS: 輕度 (MILD)";
      else if (currentVal <= 80) chiStatus = "🟠 SYSTEM_STATUS: 中度 (MODERATE)";
      else chiStatus = "🔴 SYSTEM_STATUS: 高危 (SEVERE)";

      label2.text = chiStatus;
      label2.fill = targetColor;
      label.fill = targetColor;
    }});
    }});
    </script>
</body>
</html>
"""

    with col_result:
        st.markdown("#### 🎯 即時壓力評估儀表")
        components.html(html_code, height=520)
