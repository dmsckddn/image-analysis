import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import rawpy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from scipy.optimize import curve_fit
import os


class EquilibriumAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Smartphone Analyzer (RGB Analysis)")
        self.root.geometry("1200x950")

        self.dark_file = ""
        self.blank_files = []
        self.sample_files = []
        self.input_vars = []

        self.setup_ui()

    def setup_ui(self):
        frame_files = tk.LabelFrame(
            self.root, text="1. 실험 파일 설정 (DNG)", padx=10, pady=10
        )
        frame_files.pack(fill="x", padx=10, pady=5)

        tk.Button(
            frame_files, text="다크 프레임 선택", command=self.load_dark
        ).grid(row=0, column=0, sticky="w")

        self.lbl_dark = tk.Label(frame_files, text="선택된 다크 파일 없음")
        self.lbl_dark.grid(row=0, column=1, padx=10, sticky="w")

        tk.Button(
            frame_files,
            text="바탕 용액(Blank)들 선택",
            command=self.load_blank
        ).grid(row=1, column=0, sticky="w", pady=5)

        self.lbl_blank = tk.Label(frame_files, text="선택된 바탕 파일 없음")
        self.lbl_blank.grid(row=1, column=1, padx=10, sticky="w")

        frame_samples = tk.LabelFrame(
            self.root, text="2. 샘플 리스트 (농도 입력)", padx=10, pady=10
        )
        frame_samples.pack(fill="both", expand=True, padx=10, pady=5)

        tk.Button(
            frame_samples,
            text="같은 농도의 샘플 파일들 추가",
            command=self.load_samples
        ).pack(anchor="w")

        self.canvas = tk.Canvas(frame_samples)
        scrollbar = tk.Scrollbar(
            frame_samples, orient="vertical", command=self.canvas.yview
        )

        self.scrollable_frame = tk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw"
        )
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True, pady=5)
        scrollbar.pack(side="right", fill="y")

        tk.Button(
            self.root,
            text="분석 시작",
            command=self.run_analysis,
            bg="#d32f2f",
            fg="white",
            font=("Arial", 12, "bold")
        ).pack(pady=10)

        self.frame_plot = tk.Frame(self.root)
        self.frame_plot.pack(fill="both", expand=True, padx=10, pady=5)

    def load_dark(self):
        self.dark_file = filedialog.askopenfilename(
            title="Dark DNG 선택",
            filetypes=[("DNG files", "*.dng")]
        )
        if self.dark_file:
            self.lbl_dark.config(text=os.path.basename(self.dark_file))

    def load_blank(self):
        files = filedialog.askopenfilenames(
            title="Blank DNG들 선택",
            filetypes=[("DNG files", "*.dng")]
        )
        if files:
            self.blank_files = list(files)
            self.lbl_blank.config(
                text=f"{len(self.blank_files)}개의 바탕 파일 선택됨"
            )

    def load_samples(self):
        files = filedialog.askopenfilenames(
            title="같은 농도의 샘플 DNG들을 선택하세요",
            filetypes=[("DNG files", "*.dng")]
        )
        if not files:
            return

        concentration = simpledialog.askfloat(
            "농도 입력",
            f"선택한 {len(files)}개 이미지의 공통 농도를 입력하세요.",
            parent=self.root
        )
        if concentration is None:
            return

        # 기존 목록을 유지하며 새 파일들에 공통 농도를 입력합니다.
        start_row = len(self.sample_files)
        for i, f in enumerate(files, start=start_row):
            self.sample_files.append(f)

            tk.Label(
                self.scrollable_frame, text=os.path.basename(f)
            ).grid(row=i, column=0)

            v = tk.StringVar(value=str(concentration))
            tk.Entry(
                self.scrollable_frame, textvariable=v
            ).grid(row=i, column=1)

            self.input_vars.append(v)

    def get_linear_data(self, file_path):
        with rawpy.imread(file_path) as raw:
            return raw.postprocess(
                use_camera_wb=False,
                no_auto_bright=True,
                bright=1.0,
                gamma=(1, 1),
                output_bps=16
            )

    def find_centroid(self, img_rgb):
        gray = cv2.cvtColor(
            (img_rgb / 256).astype(np.uint8), cv2.COLOR_RGB2GRAY
        )
        blurred = cv2.GaussianBlur(gray, (51, 51), 0)
        _, max_val, _, _ = cv2.minMaxLoc(blurred)
        thresh_val = max(max_val * 0.9, 10)
        _, thresh = cv2.threshold(
            blurred, thresh_val, 255, cv2.THRESH_BINARY
        )
        M = cv2.moments(thresh)

        if M["m00"] != 0:
            return int(M["m01"] / M["m00"]), int(M["m10"] / M["m00"])

        return cv2.minMaxLoc(blurred)[3][::-1]

    def save_image_safe(self, file_path, img_bgr):
        result, n = cv2.imencode(".png", img_bgr)
        if result:
            with open(file_path, mode="wb") as f:
                n.tofile(f)

    def run_analysis(self):
        if not (self.dark_file and self.blank_files and self.sample_files):
            messagebox.showerror("오류", "모든 파일을 설정하세요.")
            return

        save_dir = filedialog.askdirectory(title="결과 저장 폴더")
        if not save_dir:
            return

        # 1. Dark 분석
        dark_img = self.get_linear_data(self.dark_file)
        dy, dx = self.find_centroid(dark_img)

        dark_r = np.mean(
            dark_img[max(0, dy-25):dy+25, max(0, dx-25):dx+25, 0]
        )
        dark_g = np.mean(
            dark_img[max(0, dy-25):dy+25, max(0, dx-25):dx+25, 1]
        )
        dark_b = np.mean(
            dark_img[max(0, dy-25):dy+25, max(0, dx-25):dx+25, 2]
        )

        # RGB 평균 추출 후 이미지 메모리 해제
        del dark_img

        # 2. Multi-Blank 평균 분석 (I0 계산용)
        all_blank_r, all_blank_g, all_blank_b = [], [], []

        for bf in self.blank_files:
            b_img = self.get_linear_data(bf)
            by, bx = self.find_centroid(b_img)

            all_blank_r.append(np.mean(
                b_img[max(0, by-25):by+25, max(0, bx-25):bx+25, 0]
            ))
            all_blank_g.append(np.mean(
                b_img[max(0, by-25):by+25, max(0, bx-25):bx+25, 1]
            ))
            all_blank_b.append(np.mean(
                b_img[max(0, by-25):by+25, max(0, bx-25):bx+25, 2]
            ))

            # 다음 Blank를 읽기 전에 이미지 메모리 해제
            del b_img

        avg_blank_r = np.mean(all_blank_r)
        avg_blank_g = np.mean(all_blank_g)
        avg_blank_b = np.mean(all_blank_b)

        # Blank의 I_net (즉, Beer-Lambert의 I0)
        i0_r = avg_blank_r - dark_r
        i0_g = avg_blank_g - dark_g
        i0_b = avg_blank_b - dark_b

        # 3. 개별 샘플 분석
        detailed_data = []

        for i, f in enumerate(self.sample_files):
            try:
                conc_str = self.input_vars[i].get()
                conc = float(conc_str) if conc_str else 0.0
                img = self.get_linear_data(f)
                cy, cx = self.find_centroid(img)

                # Sample Raw 값 추출
                sample_r_raw = np.mean(
                    img[max(0, cy-25):cy+25, max(0, cx-25):cx+25, 0]
                )
                sample_g_raw = np.mean(
                    img[max(0, cy-25):cy+25, max(0, cx-25):cx+25, 1]
                )
                sample_b_raw = np.mean(
                    img[max(0, cy-25):cy+25, max(0, cx-25):cx+25, 2]
                )

                # I_net 계산 (Sample - Dark)
                i_r = sample_r_raw - dark_r
                i_g = sample_g_raw - dark_g
                i_b = sample_b_raw - dark_b

                # 흡광도 계산 — 원본 그대로
                abs_r = -np.log10(max(i_r, 1) / i0_r)
                abs_g = -np.log10(max(i_g, 1) / i0_g)
                abs_b = -np.log10(max(i_b, 1) / i0_b)

                detailed_data.append({
                    "File": os.path.basename(f),
                    "Concentration": conc,
                    "Absorbance_R": abs_r,
                    "Absorbance_G": abs_g,
                    "Absorbance_B": abs_b,
                    "I_NET_R": i_r,
                    "I_NET_G": i_g,
                    "I_NET_B": i_b,
                    "Sample_R_Raw": sample_r_raw,
                    "Sample_G_Raw": sample_g_raw,
                    "Sample_B_Raw": sample_b_raw,
                    "Dark_R": dark_r,
                    "Dark_G": dark_g,
                    "Dark_B": dark_b,
                    "ROI_Y": cy,
                    "ROI_X": cx
                })

                check_img = img.copy()
                cv2.rectangle(
                    check_img,
                    (cx-25, cy-25),
                    (cx+25, cy+25),
                    (65535, 0, 0),
                    15
                )
                img_name = (
                    f"ROI_Check_{conc}_"
                    f"{os.path.splitext(os.path.basename(f))[0]}.png"
                )
                self.save_image_safe(
                    os.path.join(save_dir, img_name),
                    cv2.cvtColor(
                        (check_img / 256).astype(np.uint8),
                        cv2.COLOR_RGB2BGR
                    )
                )

            except Exception as e:
                print(f"Error processing {f}: {e}")
                continue

        df_detailed = pd.DataFrame(detailed_data)
        summary = df_detailed.groupby("Concentration").agg({
            "Absorbance_R": ["mean", "std"],
            "Absorbance_G": ["mean", "std"],
            "Absorbance_B": ["mean", "std"]
        }).reset_index()

        summary.columns = [
            "Concentration",
            "Abs_R_mean", "Abs_R_std",
            "Abs_G_mean", "Abs_G_std",
            "Abs_B_mean", "Abs_B_std"
        ]
        summary = summary.fillna(0)

        # 엑셀 저장
        with pd.ExcelWriter(
            os.path.join(save_dir, "Experimental_Data_Full.xlsx")
        ) as writer:
            df_detailed.to_excel(
                writer, sheet_name="Raw_Data_Per_File", index=False
            )
            summary.to_excel(
                writer, sheet_name="Summary_By_Conc", index=False
            )

        self.plot_results(summary, save_dir)
        messagebox.showinfo(
            "완료", "분석 완료! (Dark, Raw, I_NET 포함됨)"
        )

    def plot_results(self, df, save_dir):
        for w in self.frame_plot.winfo_children():
            w.destroy()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        x = df["Concentration"].values

        def fit_func(x, a):
            return a * x

        # Red Channel
        y_r = df["Abs_R_mean"].values
        y_r_err = df["Abs_R_std"].values
        popt_r, _ = curve_fit(fit_func, x, y_r)
        r2_r = 1 - (
            np.sum((y_r - fit_func(x, *popt_r))**2)
            / np.sum((y_r - np.mean(y_r))**2)
        )
        ax1.errorbar(
            x, y_r, yerr=y_r_err, fmt="o", color="red", capsize=5
        )
        ax1.plot(
            x, fit_func(x, *popt_r), "r--",
            label=f"y={popt_r[0]:.4f}x\nR^2={r2_r:.4f}"
        )
        ax1.set_title("Red Channel")
        ax1.set_xlabel("Concentration")
        ax1.legend()

        # Green Channel
        y_g = df["Abs_G_mean"].values
        y_g_err = df["Abs_G_std"].values
        popt_g, _ = curve_fit(fit_func, x, y_g)
        r2_g = 1 - (
            np.sum((y_g - fit_func(x, *popt_g))**2)
            / np.sum((y_g - np.mean(y_g))**2)
        )
        ax2.errorbar(
            x, y_g, yerr=y_g_err, fmt="o", color="green", capsize=5
        )
        ax2.plot(
            x, fit_func(x, *popt_g), "g--",
            label=f"y={popt_g[0]:.4f}x\nR^2={r2_g:.4f}"
        )
        ax2.set_title("Green Channel")
        ax2.set_xlabel("Concentration")
        ax2.legend()

        # Blue Channel
        y_b = df["Abs_B_mean"].values
        y_b_err = df["Abs_B_std"].values
        popt_b, _ = curve_fit(fit_func, x, y_b)
        r2_b = 1 - (
            np.sum((y_b - fit_func(x, *popt_b))**2)
            / np.sum((y_b - np.mean(y_b))**2)
        )
        ax3.errorbar(
            x, y_b, yerr=y_b_err, fmt="o", color="blue", capsize=5
        )
        ax3.plot(
            x, fit_func(x, *popt_b), "b--",
            label=f"y={popt_b[0]:.4f}x\nR^2={r2_b:.4f}"
        )
        ax3.set_title("Blue Channel")
        ax3.set_xlabel("Concentration")
        ax3.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "Final_Plot_RGB.png"))

        canvas = FigureCanvasTkAgg(fig, master=self.frame_plot)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = EquilibriumAnalyzer(root)
    root.mainloop()