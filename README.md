# 🌟 Drastic Innovators 🌟

### 🆔 Team ID: A0084

### 🏫 College: Sri Eshwar College of Engineering

Welcome to the Drastic Innovators project! 🚀 We are a team of enthusiastic individuals dedicated to pushing the boundaries of innovation and technology. Our goal is to create impactful solutions that enhance user experiences and drive progress in various fields. 

---

# *Lead Gen Tool: The Future of Smart Customer Targeting & Marketing* 🎯💼

*Lead Gen Tool* is an advanced platform designed to redefine how businesses approach customer segmentation, professional contact retrieval, and campaign management. Powerful data-driven methods enable companies to engage with the right audience, boosting marketing efficiency and conversion success.

## *Why Lead Gen Tool?* 🤔
![image](Images/Comp.png)
In the age of data overload, companies struggle to cut through the noise and find their ideal customers. Marketing strategies are often hampered by outdated or inefficient methods of reaching target audiences, resulting in poor engagement and wasted efforts. *Lead Gen Tool* tackles this problem head-on with AI-enhanced solutions for better targeting, precision, and automation.

---

## *Challenges Faced by Modern Businesses* ❌

1. **Inefficient Customer Segmentation:** Businesses spend too much time and effort trying to group their audience manually, often missing crucial segments that can drive growth.
   
2. **Lack of Accurate Contact Information:** Outdated or incorrect contact details make it difficult to reach decision-makers, causing opportunities to slip away.

3. **Manual Campaign Management:** Marketing teams rely on generic email campaigns and lack real-time feedback, leading to poor engagement and wasted resources.

---

# 💻✨ UI Design Preview

Check out our latest UI/UX design created using Uizard! 🎨🖼️

[🚀 **View the UI Design Here**](https://app.uizard.io/p/548942ed)

We aim to provide an intuitive and beautiful user experience with our new design. Your feedback is welcome! 😃💬

---

## *Lead Gen Tool's Solution* ✅

### *DEMO* 
https://github.com/user-attachments/assets/1f8619ba-6ee2-489f-b6e5-c558974261c4

By blending the power of *AI* and *Large Language Models (LLMs), **Lead Gen Tool* solves these issues, offering businesses a comprehensive platform with the following key features:

---

### *1. AI-Powered Customer Segmentation* 📊✨

Lead Gen Tool’s customer segmentation module leverages cutting-edge AI to help businesses target customers with greater precision.

- **Industry-Based Search**: Instantly search for companies within specific industries, like Electronics, Food, or Healthcare, for tailored targeting.
- **Advanced Filters**: Refine your audience by applying filters such as company size, revenue, location, and more to focus your efforts where it matters most.
- **Web Scraping for Real-Time Data**: Automatically gather and organize customer data from public sources, keeping your customer profiles up to date.
- **AI-Powered Analysis**: Our algorithms segment and analyze customer data with high accuracy, ensuring you're targeting the right market segments effortlessly.

---

### *2. Verified Professional Contact Retrieval* 📧🔎

Obtaining accurate contact information is critical for effective outreach. Lead Gen Tool helps businesses retrieve verified contact details from target companies, focusing on key decision-makers.

- **Contact Information Extraction**: Extract essential information such as email addresses, phone numbers, and job titles from platforms like LinkedIn.
- **Role-Based Segmentation**: Focus on high-level professionals (e.g., C-suite, directors, managers), filtering out unnecessary contacts to streamline outreach.
- **AI Verification**: Ensure the accuracy of contact data using AI-powered tools that cross-reference information across various sources to provide verified results.

---

### *3. AI-Driven Campaign Management* 📈✉

Campaign management becomes effortless with Lead Gen Tool’s AI-driven features that automate personalization and provide real-time performance feedback.

- **Personalized Email Templates**: Generate dynamic email templates tailored to your recipient's role, industry, and company details.
- **Automated Campaign Execution**: Launch, track, and optimize your marketing campaigns directly from the platform, powered by AI that refines each campaign in real-time.
- **Performance Metrics & Feedback**: Monitor campaign success with detailed analytics such as open rates, click-through rates, and response tracking, all optimized for continuous improvement.

---

### Detailed Lead Gen Tool Process Flow:

![image](https://github.com/user-attachments/assets/611071cd-2f14-43fe-a7cc-3b8052c2c800)


----


### Tech Tools :

![image](https://github.com/user-attachments/assets/567a8d3c-20a4-46d6-b735-abe9720faee4)

### Intel® Toolkits:
![image](https://github.com/user-attachments/assets/8ff159b0-a560-40d5-9f10-792dd9cf9d35)

#### Intel Toolkits used:

![image](https://github.com/user-attachments/assets/1909c3a6-0e02-4f9b-9e5a-fb72cb4120da)

**Model Optimization**:
- [Intel® Extension for Pytorch](https://github.com/intel/intel-extension-for-pytorch): This tool optimizes deep learning models built in PyTorch, enabling faster training and inference on Intel® hardware such as GPUs and TPUs.
- [Intel® Ipex](https://github.com/intel/intel-extension-for-pytorch): (Intel® Extension for PyTorch): Specifically designed to optimize performance further for Intel architectures, enhancing both training and inference speed.

**Model Inference**:
- [Intel® OpenVINO™ Toolkit](https://github.com/openvinotoolkit/openvino): This toolkit is focused on optimizing inference, converting trained models into a format that is highly efficient for deployment across various Intel hardware, ensuring low-latency predictions.

**Data Analytics**:
- [Intel® Distribution for Modin](https://github.com/modin-project/modin): Used to accelerate data analysis and Exploratory Data Analysis (EDA) tasks, especially for handling large datasets, by providing better performance than traditional Pandas on Intel CPUs.

**Microsoft/phi-2**: Shown as a large-scale model that benefits from these Intel optimizations, driving efficient and faster AI model training and deployment.

### Intel ToolKits Usage Tutorial

**Install Intel's PyTorch for Optimization**
```bash
conda install libuv
python -m pip install torch==2.3.1+cxx11.abi torchvision==0.18.1+cxx11.abi torchaudio==2.3.1+cxx11.abi intel-extension-for-pytorch==2.3.110+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/mtl/us/
```

**Load and Optimize a Pre-Trained Model**
```python
#import the required libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import intel_extension_for_pytorch as ipex

model_name = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = ipex.optimize(model)
```

**Install Modin for Data Processing**
```bash
pip install modin[dask]
```

```python
import modin.pandas as pd
os.environ["MODIN_ENGINE"] = "dask"

df = pd.read_csv(file_path)
```

**Install Intel’s Optimum and Neural Compressor**
```bash
pip install optimum[openvino,nncf]
pip install "optimum-intel[extras]"@git+https://github.com/huggingface/optimum-intel.git
pip install neural-compressor[pt]
```

**Load and Optimize your Model**
```python
from optimum.intel import OVModelForCausalLM

model_id = "microsoft/phi-2"
model = OVModelForCausalLM.from_pretrained(model_id, export=True)
model.save_pretrained("ov_model")
```

---

# Benchmark Results with Intel® oneAPI Toolkits

![image](https://github.com/user-attachments/assets/df0bb37a-aa31-47bd-b56b-754915d2e87d)

1. **Model Optimization**:
   - The **Intel Neural Compressor**: This component is used to quantize the phi-2 model, which means converting a model to a more efficient format that uses less memory and increases inference speed without sacrificing accuracy. It supports both static quantization and static smooth quantization for more advanced performance improvements.
   
2. **Inference**:
   - The optimized model is passed to the **IPEX backend** for inference. The Intel® Extension for PyTorch (IPEX) backend further improves inference time by integrating the quantized model into Intel architectures.
   - **FP32 format**: This format represents the 32-bit floating point precision, indicating the model is optimized for high-precision, accurate inference.

3. **Intel OpenVINO™ Toolkit**: Further optimizes the inference of the model after compression. The OpenVINO™ toolkit supports deep learning inference across Intel CPUs, GPUs, and other accelerators to speed up model predictions.

4. **Intel Distribution for Modin**: Finally, the large datasets involved in this machine learning pipeline are managed using Intel's Modin distribution, which accelerates data processing, retrieval, and insertion tasks. It shows the efficient handling of large datasets that are crucial for AI model training and inference, enabling faster analysis with Pandas-compatible APIs.

---

![Open_VINO_Comparions](Images/open_vino_Performance.png)

#### Intel® Distribution for Modin vs Pandas

![image](https://github.com/user-attachments/assets/9426cdf3-b3b9-4368-bcc5-8f9076c99348)

**Intel® Modin vs. Pandas**: Visualizing a significant performance boost in data analysis using Intel® Modin, the image could display a bar graph comparing time taken to process large datasets using Modin versus Pandas.

**Inference & Training Performance**: Another comparison could illustrate the performance improvement achieved with Intel® Extension for PyTorch, focusing on faster training and inference times when using Intel® hardware optimizations.


Our application utilizes the following technologies:

# Tech Stack

## Frontend
<table>
  <tr>
    <td align="center" width="200">
      <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/html5/html5-original.svg" alt="HTML" width="50" height="50"/><br />
      <b>HTML</b>
    </td>
    <td align="center" width="200">
      <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/css3/css3-original.svg" alt="CSS" width="50" height="50"/><br />
      <b>CSS</b>
    </td>
    <td align="center" width="200">
      <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/javascript/javascript-original.svg" alt="JavaScript" width="50" height="50"/><br />
      <b>JavaScript</b>
    </td>
  </tr>
</table>

## Backend
<table>
  <tr>
    <td align="center" width="200">
      <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/flask/flask-original.svg" alt="Flask" width="50" height="50"/><br />
      <b>Flask</b>
    </td>
    <td align="center" width="200">
      <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="Python" width="50" height="50"/><br />
      <b>Python</b>
    </td>
  </tr>
</table>

## AI & Machine Learning
<table>
  <tr>
    <td align="center" width="200">
      <img src="https://github.com/logabaalan777/images/blob/main/assets/microsoft-phi2.jpeg" alt="phi-2" width="50" height="50"/><br />
      <b>phi-2 model</b>
    </td>
    <td align="center" width="200">
      <img src="https://github.com/logabaalan777/images/blob/main/assets/lg.png" alt="Langchain" width="50" height="50"/><br />
      <b>Langchain</b>
    </td>
    <td align="center" width="200">
      <img src="https://github.com/logabaalan777/images/blob/main/assets/logo-oneapi.png" alt="Intel OneAPI" width="50" height="50"/><br />
      <b>Intel One API</b>
    </td>
  </tr>
</table>

## Web Automation & Testing
<table>
  <tr>
    <td align="center" width="200">
      <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/selenium/selenium-original.svg" alt="Selenium" width="50" height="50"/><br />
      <b>Selenium</b>
    </td>
  </tr>
</table>

----

# <- Lead Gen Tool -> 


![image1](https://github.com/user-attachments/assets/f6cf4cfd-078e-4afe-9864-2924b1d3de34)

![image2](https://github.com/user-attachments/assets/dc2679af-bccd-4ce0-a6f5-3223bc2328c9)

![image3](https://github.com/user-attachments/assets/79d4872e-5233-4fea-87bc-948e2941081a)

![Screenshot 2024-09-13 213859](https://github.com/user-attachments/assets/75b6fae2-a1b9-4500-a00c-8523f92f47d9)

### Here is a sample data of company information.

![image](https://github.com/user-attachments/assets/eb6d3c26-8555-46b3-94f4-25cb297b564a)

----

# Resource 

# 🚀 You can view the presentation:

Please go to PPT 🎉 https://www.canva.com/design/DAGSmf3iKbU/aKbvpV_XXANb6iHHDJ2psA/edit?utm_content=DAGSmf3iKbU&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton for more details

# 🚀 Intel DevMesh Project

Check out our project on Intel DevMesh! 🌐🔗

🛠️ **View the Intel DevMesh Project Here** https://devmesh.intel.com/projects/lead-gen-tool 

----

# What we learned ![image](https://user-images.githubusercontent.com/72274851/218499685-e8d445fc-e35e-4ab5-abc1-c32462592603.png)

![image](https://github.com/user-attachments/assets/ecd13baa-a872-438b-a355-a13f3d08fefc)


✅ **Utilizing the Intel® AI Analytics Toolkit**: By utilizing the Intel® AI Analytics Toolkit, developers can leverage familiar Python* tools and frameworks to accelerate the entire data science and analytics process on Intel® architecture. This toolkit incorporates oneAPI libraries for optimized low-level computations, ensuring maximum performance from data preprocessing to deep learning and machine learning tasks. Additionally, it facilitates efficient model development through interoperability.

✅ **Seamless Adaptability**: The Intel® AI Analytics Toolkit enables smooth integration with machine learning and deep learning workloads, requiring minimal modifications.

✅ **Fostered Collaboration**: The development of such an application likely involved collaboration with a team comprising experts from diverse fields, including deep learning and data analysis. This experience likely emphasized the significance of collaborative efforts in attaining shared objectives.


<a name="About-Us"></a>
## About Us 👨‍💻🌐
Meet the talented team behind **Lead Gen Tool**:

- **Karthikeyan M**: [LinkedIn](https://www.linkedin.com/in/karthikeyan-m30112004/) | [GitHub](https://github.com/KarthikeyanM3011)
- **Barath Raj P**: [LinkedIn](https://www.linkedin.com/in/barathrajp/) | [GitHub](https://github.com/Barathaj)
- **Arun Kumar R**: [LinkedIn](https://www.linkedin.com/in/arun-kumar-99b841255/) | [GitHub](https://github.com/ArunKumar200510)
- **LogaBaalan RS**:[LinkedIn](https://www.linkedin.com/in/logabaalan-r-s-94ba82259/) | [GitHub](https://github.com/logabaalan777)

Contact us for collaborations, queries, or more information!

---

## *Unlock the Power of Smart Marketing Today* 🔑

*Lead Gen Tool* is the all-in-one platform to take your marketing efforts to the next level, powered by AI and tailored to deliver results. Say goodbye to outdated marketing methods and embrace the future with smarter, data-driven tools that ensure every outreach effort hits the mark.

*Join Lead Gen Tool today and transform your customer segmentation, contact retrieval, and campaign management with cutting-edge technology!*
es *AI and LLMs* to empower businesses to find and target their ideal customers with advanced precision.
