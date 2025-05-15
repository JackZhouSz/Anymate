import os
import gradio as gr

from Anymate.args import ui_args
from Anymate.utils.ui_utils import process_input, vis_joint, vis_connectivity, vis_skinning, prepare_blender_file
from Anymate.utils.ui_utils import get_model, get_result_joint, get_result_connectivity, get_result_skinning, get_all_models, get_all_results

with gr.Blocks() as demo:
    gr.Markdown("""
    # Anymate: Auto-rigging 3D Objects
    Welcome to the interactive demo for **[Anymate](https://anymate3d.github.io/)**, a breakthrough in automatic 3D object rigging!   
    This demo is based on our research paper: 📄 **[Anymate: A Dataset and Baselines for Learning 3D Object Rigging](https://arxiv.org/abs/2505.06227)**  
    🚀 **How It Works**
    1. **Upload** your 3D mesh (`.obj` or `.glb` format) or select a sample object from the gallery.  
    2. Click **"Load All Models"** and **"Run All Models"** to automatically generate rigging.  
    3. **Download** your animatable object as a `.blender` file—ready for animation!  
                
    ✨ No manual rigging, no hassle—just drag, drop, and animate.  
    📌 *Tip: Check out our [project page](https://anymate3d.github.io/) for datasets, code, and more!*  
    """)

    pc = gr.State(value=None)
    normalized_mesh_file = gr.State(value=None)

    result_joint = gr.State(value=None)
    result_connectivity = gr.State(value=None)
    result_skinning = gr.State(value=None)

    model_joint = gr.State(value=None)
    model_connectivity = gr.State(value=None)
    model_skinning = gr.State(value=None)
    
    with gr.Row():
        with gr.Column(scale=0.33):
            # Input section
            gr.Markdown("## Input")
            mesh_input = gr.Model3D(label="Input 3D Mesh", clear_color=[0.0, 0.0, 0.0, 0.0])

        with gr.Column(scale=0.67):
            # Sample 3D objects section
            gr.Markdown("## Select to load sample objects")
            sample_objects_dir = './samples'
            sample_objects = [os.path.join(sample_objects_dir, f) for f in os.listdir(sample_objects_dir) 
                            if f.endswith('.png') and os.path.isfile(os.path.join(sample_objects_dir, f))]
            sample_objects.sort()
            print(len(sample_objects))
            
            sample_gallery = gr.Gallery(
                label="Sample Objects",
                value=sample_objects,
                columns=5,
                object_fit="contain",
                height="200px",
                allow_preview=False
            )
    # Checkpoint paths
    joint_models_dir = 'Anymate/checkpoints/joint'
    joint_models = [os.path.join(joint_models_dir, f) for f in os.listdir(joint_models_dir) 
                    if os.path.isfile(os.path.join(joint_models_dir, f))]
    with gr.Row():
        joint_checkpoint = gr.Dropdown(
            label="Joint Checkpoint",
            choices=joint_models,
            value=ui_args.checkpoint_joint,
            interactive=True
        )
        joint_status = gr.Checkbox(label="Joint Model Status", value=False, interactive=False, scale=0.3)
        # with gr.Column():
        #     with gr.Row():
        #         load_joint_btn = gr.Button("Load", scale=0.3)
                
        #     process_joint_btn = gr.Button("Process", scale=0.3)

    conn_models_dir = 'Anymate/checkpoints/conn' 
    conn_models = [os.path.join(conn_models_dir, f) for f in os.listdir(conn_models_dir)
                    if os.path.isfile(os.path.join(conn_models_dir, f))]
    with gr.Row():
        conn_checkpoint = gr.Dropdown(
            label="Connection Checkpoint",
            choices=conn_models,
            value=ui_args.checkpoint_conn,
            interactive=True
        )
        conn_status = gr.Checkbox(label="Connectivity Model Status", value=False, interactive=False, scale=0.3)
        # with gr.Column():
        #     with gr.Row():
        #         load_conn_btn = gr.Button("Load", scale=0.3)
                
        #     process_conn_btn = gr.Button("Process", scale=0.3)

    skin_models_dir = 'Anymate/checkpoints/skin'
    skin_models = [os.path.join(skin_models_dir, f) for f in os.listdir(skin_models_dir)
                    if os.path.isfile(os.path.join(skin_models_dir, f))]
    with gr.Row():
        skin_checkpoint = gr.Dropdown(
            label="Skin Checkpoint", 
            choices=skin_models,
            value=ui_args.checkpoint_skin,
            interactive=True
        )
        skin_status = gr.Checkbox(label="Skinning Model Status", value=False, interactive=False, scale=0.3)
        # with gr.Column():
        #     with gr.Row():
        #         load_skin_btn = gr.Button("Load", scale=0.3)
               
        #     process_skin_btn = gr.Button("Process", scale=0.3)

    
    with gr.Row():
        # Parameters for DBSCAN clustering algorithm used to adjust joint clustering
        eps = gr.Number(label="Epsilon", value=0.03, interactive=True, info="Controls the maximum distance between joints in a cluster")
        min_samples = gr.Number(label="Min Samples", value=1, interactive=True, info="Minimum number of joints required to form a cluster")

    with gr.Row():
        load_all_btn = gr.Button("Load all models", scale=1)
        process_all_btn = gr.Button("Run all models", scale=1)
        # download_btn = gr.DownloadButton("Blender File Not Ready", scale=0.3)
        # blender_file = gr.File(label="Blender File", scale=1)

    with gr.Row():
        gr.Markdown("## Output")
    with gr.Row():
        with gr.Column():
            # Output section
            gr.Markdown("#### (wireframe display mode)")
            mesh_output = gr.Model3D(label="Output 3D Mesh", clear_color=[0.0, 0.0, 0.0, 0.0], display_mode="wireframe")

        with gr.Column():
            # Output section
            gr.Markdown("#### (solid display mode)")
            mesh_output2 = gr.Model3D(label="Output 3D Mesh", clear_color=[0.0, 0.0, 0.0, 0.0], display_mode="solid")
        with gr.Column():
            # Output section
            gr.Markdown("#### (blender file)")
            blender_file = gr.File(label="Output Blender File")
    
    mesh_input.change(
        process_input,
        inputs=mesh_input,
        outputs=[normalized_mesh_file, mesh_output, mesh_output2, blender_file, pc, result_joint, result_connectivity, result_skinning]
    )

    def get_select_object(evt: gr.SelectData):
        return f'./samples/sample{evt.index+1}.obj'
    sample_gallery.select(
        fn=get_select_object,
        inputs=None,
        outputs=mesh_input
    ).then(
        process_input,
        inputs=mesh_input,
        outputs=[normalized_mesh_file, mesh_output, mesh_output2, blender_file, pc, result_joint, result_connectivity, result_skinning]
    )

    normalized_mesh_file.change(
        lambda x: x,
        inputs=normalized_mesh_file,
        outputs=mesh_input
    )

    result_joint.change(
        vis_joint,
        inputs=[normalized_mesh_file, result_joint],
        outputs=[mesh_output, mesh_output2]
    )

    result_connectivity.change(
        vis_connectivity,
        inputs=[normalized_mesh_file, result_joint, result_connectivity],
        outputs=[mesh_output, mesh_output2]
    )

    result_skinning.change(
        vis_skinning,
        inputs=[normalized_mesh_file, result_joint, result_connectivity, result_skinning],
        outputs=[mesh_output, mesh_output2]
    )

    result_skinning.change(
        prepare_blender_file,
        inputs=[normalized_mesh_file],
        outputs=blender_file
    )

    joint_checkpoint.change(
        get_model,
        inputs=joint_checkpoint,
        outputs=[model_joint, joint_status]
    )
    
    conn_checkpoint.change(
        get_model,
        inputs=conn_checkpoint,
        outputs=[model_connectivity, conn_status]
    )
    
    skin_checkpoint.change(
        get_model,
        inputs=skin_checkpoint,
        outputs=[model_skinning, skin_status]
    )

    load_all_btn.click(
        get_all_models,
        inputs=[joint_checkpoint, conn_checkpoint, skin_checkpoint],
        outputs=[model_joint, model_connectivity, model_skinning, joint_status, conn_status, skin_status]
    )

    process_all_btn.click(
        get_all_results,
        inputs=[normalized_mesh_file, model_joint, model_connectivity, model_skinning, pc, eps, min_samples],
        outputs=[result_joint, result_connectivity, result_skinning]
    )

    # load_joint_btn.click(
    #     fn=get_model,
    #     inputs=joint_checkpoint,
    #     outputs=[model_joint, joint_status]
    # )

    # load_conn_btn.click(
    #     fn=get_model,
    #     inputs=conn_checkpoint,
    #     outputs=[model_connectivity, conn_status]
    # )

    # load_skin_btn.click(
    #     fn=get_model,
    #     inputs=skin_checkpoint,
    #     outputs=[model_skinning, skin_status]
    # )

    # process_joint_btn.click(
    #     fn=get_result_joint,
    #     inputs=[normalized_mesh_file, model_joint, pc, eps, min_samples],
    #     outputs=result_joint
    # )
    
    # process_conn_btn.click(
    #     fn=get_result_connectivity,
    #     inputs=[normalized_mesh_file, model_connectivity, pc, result_joint],
    #     outputs=result_connectivity
    # )
    
    # process_skin_btn.click(
    #     fn=get_result_skinning,
    #     inputs=[normalized_mesh_file, model_skinning, pc, result_joint, result_connectivity],
    #     outputs=result_skinning
    # )


demo.launch(debug=True, inline=True)
