# --- Import necessary libraries ---
import streamlit as st
from PIL import Image
import os

# --- Main function for the Streamlit tab ---
def visual_guide_tab():
    st.title("👁️ Visual Diagnosis Guide")
    st.write("This guide provides example images and key medical indicators for each of the 5 stages of Diabetic Retinopathy.")
    st.info("Understanding these visual cues is key to know how  classifications are made.")
    st.markdown("---")

    # --- Define the local path for our guide images ---
    # The app will look for a folder named 'guide_images' in the same directory as app.py
    GUIDE_IMAGES_PATH = "guide_images"

    # --- We store the information for each class in a dictionary ---
    # The 'images' list now contains local filenames instead of web URLs.
    class_info = {
        "0 - No DR": {
            "description": "No Diabetic Retinopathy - Normal Fundus. \n\nDescription & Significance: The retina appears healthy with no clinically detectable lesions (microaneurysms, hemorrhages, or exudates). This confirms that your systemic diabetes management is successfully protecting the microvasculature of your eyes. \n\nAction & Advice: This is the time for PREVENTION. Continue rigorous control of your A1C (target usually <7%), blood pressure (<130/80 mmHg), and cholesterol/lipids. Adhere to your advised remedy, diet, and exercise program diligently. Quitting smoking is paramount, as it accelerates vascular damage. \n\nMonitoring Duration: A repeat comprehensive, dilated fundus exam is recommended in 1 year. Report any new visual symptoms immediately. \n\nSpecial Note: Even with a normal fundus, a Type 1 Diabetic should have a baseline exam within 5 years of diagnosis; a Type 2 Diabetic should have a baseline exam at the time of diagnosis.",
            "folder": "0",
            "images": ["0_1.jpg", "0_2.jpg", "0_3.jpg"]
        },
        "1 - Mild DR": {
            "description":"Mild Diabetic Retinopathy (Mild NPDR). \n\n*Description & Significance:* The earliest signs of vascular damage, typically the presence of only *microaneurysms* (tiny bulges in the capillary walls). Vision is usually unaffected. This is the *critical window* for preventing progression. \n\n*Action & Advice:* The primary management is *SYSTEMIC CONTROL. Work closely with your endocrinologist/PCP to achieve optimal metabolic targets. Consider an initial work-up, potentially including a **Fundus Fluorescein Angiogram (FFA)* or *Optical Coherence Tomography (OCT)* if Macular Edema (DME) is suspected, though treatment is rarely needed at this stage without DME. Ask your doctor about the potential benefit of medications like *Fenofibrate, which has shown to reduce the risk of progression in some patients with NPDR. \n\nMonitoring Duration:* Repeat dilated eye exam every *6 to 12 months. Close monitoring is essential as approximately 16% of Mild NPDR cases can progress to Severe NPDR within 4 years. \n\nSpecial Note:* No immediate referral needed for the retinopathy itself, but urgent referral is required if *Diabetic Macular Edema (DME)* is diagnosed.",
            "folder": "1",
            "images": ["1_1.jpg", "1_2.jpg", "1_3.jpg"]
        },
        "2 - Moderate DR": {
            "description": "Moderate Diabetic Retinopathy (Moderate NPDR). \n\n*Description & Significance:* Increased number of microaneurysms, *dot-and-blot hemorrhages, and possibly **hard exudates* or *cotton wool spots. The risk of vision-threatening complications like DME is significantly higher. \n\nAction & Advice:* *Refer to a specialist (Retina Specialist/Ophthalmologist). The specialist will prioritize ruling out or treating **Clinically Significant Macular Edema (CSME), which is the main cause of vision loss at this stage. Treatment for DME typically involves **Intravitreal Anti-VEGF Injections (e.g., Avastin, Lucentis, Eylea)* or *Focal/Grid Laser Photocoagulation. Systemic control (A1C, BP) must be aggressively tightened. \n\nMonitoring Duration:* Follow-up with the specialist every *3 to 6 months. The 5-year progression rate to Proliferative Diabetic Retinopathy (PDR) can be as high as 50% without intervention. \n\nSpecial Note:* Treatment of DME is generally the first intervention. In some cases, Anti-VEGF injections may be considered even without DME to regress retinopathy severity.",
            "folder": "2",
            "images": ["2_1.jpg", "2_2.jpg", "2_3.jpg"]
        },
        "3 - Severe DR": {
            "description":"Severe Diabetic Retinopathy (Severe NPDR). \n\n*Description & Significance:* This stage meets the 4-2-1 rule (Severe Hemorrhages/Microaneurysms in 4 quadrants, *Venous Beading* in 2 quadrants, *Intraretinal Microvascular Abnormalities (IRMA)* in 1 quadrant). The retina is deprived of oxygen (*ischemia), leading to a high output of **Vascular Endothelial Growth Factor (VEGF), which signals the body to grow new, abnormal vessels (PDR). \n\nAction & Advice:* *URGENT referral to a Retina Specialist.* Aggressive treatment is necessary. The specialist will likely initiate *Panretinal Photocoagulation (PRP) laser surgery* to ablate the ischemic peripheral retina and reduce VEGF drive. Alternatively, or in combination, *Anti-VEGF injections* may be started as they can cause rapid regression of severe retinopathy features and delay progression to PDR. Systemic control must be optimized immediately. \n\n*Monitoring Duration:* Close follow-up every *2 to 4 months* is mandatory, as approximately 50% of Severe NPDR cases progress to PDR within one year. \n\n*Special Note:* Treatment is aimed at preventing the sight-threatening complications of PDR and is often initiated before PDR officially develops.",
            "folder": "3",
            "images": ["3_1.jpg", "3_2.jpg", "3_3.jpg"]
        },
        "4 - Proliferative DR": {
            "description":"Proliferative Diabetic Retinopathy (PDR). \n\n*Description & Significance:* The most advanced stage, characterized by *Neovascularization* (abnormal, fragile new blood vessels growing on the optic disc (*NVD) or elsewhere on the retina (NVE)). These vessels are prone to bleeding, causing **Vitreous Hemorrhage, or forming scar tissue that can lead to **Tractional Retinal Detachment. This is a **vision-threatening emergency. \n\nAction & Advice:* *IMMEDIATE referral to a Retina Specialist.* The standard treatment is urgent *Panretinal Photocoagulation (PRP)* laser to prevent or regress the new vessels, usually over multiple sessions. *Intravitreal Anti-VEGF injections* are often used as first-line therapy, sometimes to stabilize the eye before PRP, as they cause a faster regression of neovascularization. For complications like non-clearing Vitreous Hemorrhage or Retinal Detachment, *Vitrectomy Surgery* is necessary to remove blood and scar tissue. \n\n*Monitoring Duration:* Frequent, specialized follow-up, often weekly or monthly, until neovascularization is fully regressed and the eye is stable. \n\n*Special Note:* Patients with PDR are also at risk for *Neovascular Glaucoma* (NVG), a severe form of glaucoma caused by new vessel growth on the iris, which can lead to permanent, painful blindness.",
            "folder": "4",
            "images": ["4_1.jpg", "4_2.jpg", "4_3.jpg"]
        }
    }

    # --- Create a dropdown menu to select the class ---
    selected_class = st.selectbox("Select a Diagnosis Stage to Explore:", list(class_info.keys()))

    st.markdown("---")

    # --- Display the information for the selected class ---
    if selected_class:
        info = class_info[selected_class]
        
        st.subheader(f"Key Features of: {selected_class}")
        st.markdown(f"> {info['description']}")
        
        st.subheader("Example Images:")
        
        # Create three columns to display the images side-by-side
        cols = st.columns(3)
        try:
            for i, col in enumerate(cols):
                with col:
                    # --- CHANGE: Construct the local file path ---
                    image_path = os.path.join(GUIDE_IMAGES_PATH, info['folder'], info['images'][i])
                    
                    # Check if the file exists before trying to open it
                    if os.path.exists(image_path):
                        img = Image.open(image_path)
                        st.image(img, use_column_width=True, caption=f"Example {i+1}")
                    else:
                        st.warning(f"Image not found: {info['images'][i]}")
        except Exception as e:
            st.error(f"An error occurred while loading images: {e}")