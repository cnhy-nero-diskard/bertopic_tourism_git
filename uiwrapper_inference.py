import customtkinter as ctk
from tkinter import filedialog, messagebox
from bertopic import BERTopic
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from collections import Counter
import webbrowser
import os
import gc
import asyncio
import threading

class BERTopicUI:
    def __init__(self, root):
        ctk.set_appearance_mode("dark")  # Options: "light", "dark", "system"
        ctk.set_default_color_theme("blue")
        
        self.root = root
        self.root.title("BERTopic Analysis Tool")
        self.root.geometry("800x900")
        self.model_path = None
        self.topic_model = None
        
        # Create UI components
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        self.title_label = ctk.CTkLabel(self.root, text="BERTOPIC TOURISM (TOPIC MODEL - INFERENCE)", font=("Arial", 24, "bold"))
        self.title_label.pack(pady=20)
        
        # Model Selection
        self.model_frame = ctk.CTkFrame(self.root)
        self.model_frame.pack(pady=10, padx=10, fill="x")
        
        ctk.CTkLabel(self.model_frame, text="BERTopic Model", font=("Arial", 14)).pack(side="left", padx=10)
        self.model_label = ctk.CTkLabel(self.model_frame, text="No model selected", text_color="gray")
        self.model_label.pack(side="left", padx=10)
        ctk.CTkButton(self.model_frame, text="Load Model", command=self.load_model).pack(side="right", padx=10)
        
        # Text Input
        self.text_input = ctk.CTkTextbox(self.root, height=200)  # Increased height
        self.text_input.pack(pady=10, padx=10, fill="both")
        
        # Button Frame
        self.button_frame = ctk.CTkFrame(self.root)
        self.button_frame.pack(pady=10)
        
        self.run_button = ctk.CTkButton(self.button_frame, text="Run Analysis", command=self.run_analysis)
        self.run_button.pack(side="left", padx=10)
        
        self.export_button = ctk.CTkButton(self.button_frame, text="Export to HTML", command=self.export_visuals)
        self.export_button.pack(side="left", padx=10)
        
        # Output Section
        self.output_frame = ctk.CTkFrame(self.root)
        self.output_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        self.output_text = ctk.CTkTextbox(self.output_frame, height=200)  # Decreased height
        self.output_text.pack(fill="both", expand=True, padx=10, pady=5)
        self.output_text.configure(state="disabled")  # Make uneditable
        
        # Visualization Section
        self.figure = Figure(figsize=(15, 10))
        self.ax = None
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.output_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def load_model(self):
        path = filedialog.askopenfilename(title="Select BERTopic Model", filetypes=[("BERTopic Models", "*.*")])
        if self.topic_model is not None:
            del self.topic_model
            self.topic_model = None
            gc.collect()
        if path:
            self.model_path = path
            self.model_label.configure(text=path.split('/')[-1], text_color="green")
            try:
                self.topic_model = BERTopic.load(path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def run_analysis(self):
        asyncio.run(self.async_run_analysis())
    
    async def async_run_analysis(self):
        if not self.topic_model:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        text = self.text_input.get("1.0", "end").strip()
        if not text:
            messagebox.showerror("Error", "Please enter some text first")
            return
        
        try:
            self.output_text.configure(state="normal")  # Make editable temporarily
            self.output_text.delete("1.0", "end")
            self.figure.clear()
            self.ax = None
            
            # Split text into documents (assuming one document per line)
            docs = text.split('\n')
            
            # Perform topic inference
            loop = asyncio.get_event_loop()
            topics, probabilities = await loop.run_in_executor(None, self.topic_model.transform, docs)
            
            # Combine topics and probabilities and sort by probability in descending order
            topic_prob_pairs = sorted(zip(topics, probabilities), key=lambda x: x[1], reverse=True)
            sorted_topics, sorted_probabilities = zip(*topic_prob_pairs)
            
            # Display the results
            self.output_text.insert("end", "Document Topics and Probabilities:\n")
            for i, (topic, prob) in enumerate(zip(sorted_topics, sorted_probabilities)):
                print(f"{topic} ------------- {prob}")
                self.output_text.insert("end", f"Document {i+1}: Topic {topic}, Probability {prob:.4f}\n")
            
            # Get topic information
            topics_info = self.topic_model.get_topics()
            self.output_text.insert("end", "\nTopic Information:\n")
            self.output_text.insert("end", f"{topics_info[0]}\n")
            
            # Get all unique topics excluding -1
            unique_topics = list(set(topics))
            unique_topics = [t for t in unique_topics if t != -1]
            
            # Calculate average probability for each topic
            topic_probabilities = {}
            for topic, prob in zip(topics, probabilities):
                if topic in topic_probabilities:
                    topic_probabilities[topic] += prob
                else:
                    topic_probabilities[topic] = prob
            
            # Sort topics by average probability in descending order
            sorted_topics = sorted(topic_probabilities.keys(), key=lambda x: topic_probabilities[x], reverse=True)
            
            # Get top 10 topics based on probability
            top_10_topics = sorted_topics[:10]
            
            # Create a grid of subplots for the top 10 topics
            num_topics = len(top_10_topics)
            cols = 2
            rows = (num_topics // 2) + (1 if num_topics % 2 != 0 else 0)
            
            self.figure.set_size_inches(15, 5*rows)
            
            plt.rcParams['font.family'] = 'Noto Sans'  # or 'Noto Sans'
            plt.rcParams['axes.unicode_minus'] = False            
            for idx, topic in enumerate(top_10_topics):
                topic_words = self.topic_model.get_topic(topic)
                # Get top 5 words
                top_words = topic_words[:5]
                
                # Extract words and probabilities
                words = [word for word, prob in top_words]
                probs = [prob for word, prob in top_words]
                
                # Create subplot for each topic
                ax = self.figure.add_subplot(rows, cols, idx + 1)
                sns.barplot(x=probs, y=words, palette="viridis", ax=ax)
                
                # Calculate the average probability for the topic
                avg_prob = topic_probabilities[topic] / topics.count(topic)
                
                # Set title with probability score and color based on probability
                title_color = "green" if avg_prob > 0.5 else "red"
                ax.set_title(f"Topic {topic} (Prob: {avg_prob:.2f})", color=title_color)
                ax.set_xlabel("Probability")
                ax.set_ylabel("")
                ax.tick_params(axis='y', labelsize=8)
                
                # Set font for y-axis labels
                for tick in ax.yaxis.get_ticklabels():
                    tick.set_fontname('Malgun Gothic')
            
            # Add more spacing between subplots
            plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=3.0)
            self.canvas.draw()
            
            self.output_text.configure(state="disabled")  # Make uneditable again
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}") 

    def export_visuals(self):
        asyncio.run(self.async_export_visuals())
    
    async def async_export_visuals(self):
        if not self.figure:
            messagebox.showerror("Error", "No visualization available to export")
            return
        
        # Save the current figure
        save_path = filedialog.asksaveasfilename(
            title="Save Visualization",
            defaultextension=".html",
            filetypes=[("HTML File", "*.html")]
        )
        
        if not save_path:
            return
        
        # Create a temporary PNG file
        temp_img_path = os.path.splitext(save_path)[0] + ".png"
        plt.figure(self.figure.number)
        plt.savefig(temp_img_path, bbox_inches='tight')
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>BERTopic Visualization</title>
        </head>
        <body>
            <h1>BERTopic Visualization</h1>
            <img src="{os.path.basename(temp_img_path)}" alt="BERTopic Visualization" style="max-width: 100%">
        </body>
        </html>
        """
        
        # Save HTML file
        with open(save_path, "w") as f:
            f.write(html_content)
        
        messagebox.showinfo("Success", f"Visualization exported to {save_path}")

if __name__ == "__main__":
    root = ctk.CTk()
    app = BERTopicUI(root)
    root.mainloop()