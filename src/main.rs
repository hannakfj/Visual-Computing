extern crate nalgebra_glm as glm;
use std::{mem, ptr, os::raw::c_void};
use glutin::event::{Event, WindowEvent, DeviceEvent, KeyboardInput, ElementState::{Pressed, Released}, VirtualKeyCode::{self, *}};
use glutin::event_loop::ControlFlow;
use std::time::{Instant, Duration};
use std::sync::{Mutex, Arc, RwLock};
use std::thread;


mod shader;
mod toolbox;
mod util;
mod mesh;
mod scene_graph;
use scene_graph::SceneNode;

const INITIAL_SCREEN_W: u32 = 800;
const INITIAL_SCREEN_H: u32 = 600;

fn byte_size_of_array<T>(val: &[T]) -> isize {
    std::mem::size_of_val(&val[..]) as isize
}

fn pointer_to_array<T>(val: &[T]) -> *const c_void {
    &val[0] as *const T as *const c_void
}

fn size_of<T>() -> i32 {
    mem::size_of::<T>() as i32
}

unsafe fn create_vao(vertices: &Vec<f32>, indices: &Vec<u32>, colors: &Vec<f32>, normals: &Vec<f32>) -> u32 {
    let mut vao_id: u32 = 0;
    gl::GenVertexArrays(1, &mut vao_id);
    gl::BindVertexArray(vao_id);

    // Vertex Buffer (positions)
    let mut vbo_id: u32 = 0;
    gl::GenBuffers(1, &mut vbo_id);
    gl::BindBuffer(gl::ARRAY_BUFFER, vbo_id);
    gl::BufferData(
        gl::ARRAY_BUFFER,
        byte_size_of_array(vertices),
        pointer_to_array(vertices),
        gl::STATIC_DRAW,
    );
    gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, 3 * size_of::<f32>(), ptr::null());
    gl::EnableVertexAttribArray(0);

    // Color Buffer
    let mut cbo_id: u32 = 0;
    gl::GenBuffers(1, &mut cbo_id);
    gl::BindBuffer(gl::ARRAY_BUFFER, cbo_id);
    gl::BufferData(
        gl::ARRAY_BUFFER,
        byte_size_of_array(colors),
        pointer_to_array(colors),
        gl::STATIC_DRAW,
    );
    gl::VertexAttribPointer(1, 4, gl::FLOAT, gl::FALSE, 4 * size_of::<f32>(), ptr::null());
    gl::EnableVertexAttribArray(1);

    // Normal Buffer
    let mut nbo_id: u32 = 0;
    gl::GenBuffers(1, &mut nbo_id);
    gl::BindBuffer(gl::ARRAY_BUFFER, nbo_id);
    gl::BufferData(
        gl::ARRAY_BUFFER,
        byte_size_of_array(normals),
        pointer_to_array(normals),
        gl::STATIC_DRAW,
    );
    gl::VertexAttribPointer(2, 3, gl::FLOAT, gl::FALSE, 3 * size_of::<f32>(), ptr::null());
    gl::EnableVertexAttribArray(2);

    // Index Buffer
    let mut ibo_id: u32 = 0;
    gl::GenBuffers(1, &mut ibo_id);
    gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ibo_id);
    gl::BufferData(
        gl::ELEMENT_ARRAY_BUFFER,
        byte_size_of_array(indices),
        pointer_to_array(indices),
        gl::STATIC_DRAW,
    );

    gl::BindVertexArray(0);
    vao_id
}

fn create_helicopter_vaos(helicopter: &mesh::Helicopter) -> (u32, u32, u32, u32) {
    // Create VAOs for each part of the helicopter
    let body_vao = unsafe {
        create_vao(
            &helicopter.body.vertices,
            &helicopter.body.indices,
            &helicopter.body.colors,
            &helicopter.body.normals
        )
    };

    let door_vao = unsafe {
        create_vao(
            &helicopter.door.vertices,
            &helicopter.door.indices,
            &helicopter.door.colors,
            &helicopter.door.normals
        )
    };

    let main_rotor_vao = unsafe {
        create_vao(
            &helicopter.main_rotor.vertices,
            &helicopter.main_rotor.indices,
            &helicopter.main_rotor.colors,
            &helicopter.main_rotor.normals
        )
    };

    let tail_rotor_vao = unsafe {
        create_vao(
            &helicopter.tail_rotor.vertices,
            &helicopter.tail_rotor.indices,
            &helicopter.tail_rotor.colors,
            &helicopter.tail_rotor.normals
        )
    };

    (body_vao, door_vao, main_rotor_vao, tail_rotor_vao)
}

fn draw_helicopter(
    body_vao: u32,
    door_vao: u32,
    main_rotor_vao: u32,
    tail_rotor_vao: u32,
    helicopter: &mesh::Helicopter
) {
    // Draw the helicopter body
    unsafe {
        gl::BindVertexArray(body_vao);
        gl::DrawElements(
            gl::TRIANGLES,
            helicopter.body.index_count,
            gl::UNSIGNED_INT,
            std::ptr::null()
        );
    }

    // Draw the helicopter door
    unsafe {
        gl::BindVertexArray(door_vao);
        gl::DrawElements(
            gl::TRIANGLES,
            helicopter.door.index_count,
            gl::UNSIGNED_INT,
            std::ptr::null()
        );
    }

    // Draw the helicopter main rotor
    unsafe {
        gl::BindVertexArray(main_rotor_vao);
        gl::DrawElements(
            gl::TRIANGLES,
            helicopter.main_rotor.index_count,
            gl::UNSIGNED_INT,
            std::ptr::null()
        );
    }

    // Draw the helicopter tail rotor
    unsafe {
        gl::BindVertexArray(tail_rotor_vao);
        gl::DrawElements(
            gl::TRIANGLES,
            helicopter.tail_rotor.index_count,
            gl::UNSIGNED_INT,
            std::ptr::null()
        );
    }
}

fn main() {


    let el = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new()
        .with_title("Gloom-rs")
        .with_resizable(true)
        .with_inner_size(glutin::dpi::LogicalSize::new(INITIAL_SCREEN_W, INITIAL_SCREEN_H));
    let cb = glutin::ContextBuilder::new()
        .with_vsync(true);
    let windowed_context = cb.build_windowed(wb, &el).unwrap();

    // Set up a shared vector for keeping track of currently pressed keys
    let arc_pressed_keys = Arc::new(Mutex::new(Vec::<VirtualKeyCode>::with_capacity(10)));
    // Make a reference of this vector to send to the render thread
    let pressed_keys = Arc::clone(&arc_pressed_keys);

    // Set up shared tuple for tracking mouse movement between frames
    let arc_mouse_delta = Arc::new(Mutex::new((0f32, 0f32)));
    // Make a reference of this tuple to send to the render thread
    let mouse_delta = Arc::clone(&arc_mouse_delta);

    // Set up shared tuple for tracking changes to the window size
    let arc_window_size = Arc::new(Mutex::new((INITIAL_SCREEN_W, INITIAL_SCREEN_H, false)));
    // Make a reference of this tuple to send to the render thread
    let window_size = Arc::clone(&arc_window_size);
    

 
    // Spawn a separate thread for rendering, so event handling doesn't block rendering
    let render_thread = thread::spawn(move || {
        
        // Acquire the OpenGL Context and load the function pointers.
        // This has to be done inside of the rendering thread, because
        // an active OpenGL context cannot safely traverse a thread boundary
        let context = unsafe {
            let c = windowed_context.make_current().unwrap();
            gl::load_with(|symbol| c.get_proc_address(symbol) as *const _);
            c
        };

        let mut window_aspect_ratio = INITIAL_SCREEN_W as f32 / INITIAL_SCREEN_H as f32;
        let first_frame_time = Instant::now();
        let mut camera_pos = glm::vec3(-40.0, 10.0, 0.0);
        let mut camera_yaw = 0.0_f32;
        let mut camera_pitch = 0.0_f32;
        let mut root_node = SceneNode::new(); 

        // Set up openGL
        unsafe {
            gl::Enable(gl::DEPTH_TEST);
            gl::DepthFunc(gl::LESS);
            gl::Enable(gl::CULL_FACE);
            gl::Disable(gl::MULTISAMPLE);
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
            gl::Enable(gl::DEBUG_OUTPUT_SYNCHRONOUS);
            gl::DebugMessageCallback(Some(util::debug_callback), ptr::null());

            // Print some diagnostics
            println!("{}: {}", util::get_gl_string(gl::VENDOR), util::get_gl_string(gl::RENDERER));
            println!("OpenGL\t: {}", util::get_gl_string(gl::VERSION));
            println!("GLSL\t: {}", util::get_gl_string(gl::SHADING_LANGUAGE_VERSION));
        };

        // Load terrain mesh
        let terrain_mesh = mesh::Terrain::load("./resources/lunarsurface.obj");
        let terrain_vao = unsafe {
            create_vao(&terrain_mesh.vertices, &terrain_mesh.indices, &terrain_mesh.colors, &terrain_mesh.normals)
        };

        // Load the helicopter model
        let helicopter = mesh::Helicopter::load("./resources/helicopter.obj");

        // Create VAOs for the helicopter parts
        let (body_vao, door_vao, main_rotor_vao, tail_rotor_vao) = create_helicopter_vaos(&helicopter);

        // Create a terrain node and add it to the root node
        let mut terrain_node = SceneNode::from_vao(terrain_vao, terrain_mesh.index_count);
        unsafe {
            root_node.as_mut().add_child(&terrain_node);
        }

        // Create 5 helicopter root nodes with individual offsets
        let mut helicopter_root_nodes = Vec::new();
        let mut main_rotor_nodes = Vec::new();
        let mut tail_rotor_nodes = Vec::new();

        for i in 0..5 {
            let offset = (i as f32) * 60.0; // Use offset to prevent helicopters from colliding

            let mut helicopter_root_node = SceneNode::new();
            let mut helicopter_body_node = SceneNode::from_vao(body_vao, helicopter.body.index_count);
            let mut helicopter_door_node = SceneNode::from_vao(door_vao, helicopter.door.index_count);
            let mut main_rotor_node = SceneNode::from_vao(main_rotor_vao, helicopter.main_rotor.index_count);
            let mut tail_rotor_node = SceneNode::from_vao(tail_rotor_vao, helicopter.tail_rotor.index_count);

            // Set the reference points for the rotors
            main_rotor_node.reference_point = glm::vec3(0.0, 0.0, 0.0); // Main rotor rotates around the origin
            tail_rotor_node.reference_point = glm::vec3(0.35, 2.3, 10.4); // Tail rotor reference point

            // Adding helicopter parts to the helicopter root node
            unsafe {
                helicopter_root_node.as_mut().add_child(&helicopter_body_node);
                helicopter_root_node.as_mut().add_child(&helicopter_door_node);
                helicopter_root_node.as_mut().add_child(&main_rotor_node);
                helicopter_root_node.as_mut().add_child(&tail_rotor_node);
                terrain_node.as_mut().add_child(&helicopter_root_node); 
                
    /*             terrain_node.as_mut().add_child(&helicopter_root_node);
    */        }

            // Offset the helicopter's starting position
            helicopter_root_node.position.x += offset;
    /*       helicopter_root_node.position.y = 100.0; */

            // Store the nodes for each helicopter
            helicopter_root_nodes.push(helicopter_root_node);
            main_rotor_nodes.push(main_rotor_node);
            tail_rotor_nodes.push(tail_rotor_node);
        }

        let mut perspective_matrix: glm::Mat4 = glm::perspective(
            window_aspect_ratio,
            45.0_f32.to_radians(),
            1.0,
            1000.0,
        );
        unsafe fn draw_scene(
            node: &SceneNode,
            view_projection_matrix: &glm::Mat4,
            transformation_so_far: &glm::Mat4,
            shader_program: u32  
        ) {
            // Compute the local transformation matrix for this node
            let translation_to_reference = glm::translation(&node.reference_point);
            let rotation = glm::rotation(node.rotation.y, &glm::vec3(0.0, 1.0, 0.0))
                * glm::rotation(node.rotation.x, &glm::vec3(1.0, 0.0, 0.0))
                * glm::rotation(node.rotation.z, &glm::vec3(0.0, 0.0, 1.0));
            let translation_back_from_reference = glm::translation(&(-node.reference_point));
            let local_transform = glm::translation(&node.position)* translation_to_reference * rotation * translation_back_from_reference ;
            let model_matrix = transformation_so_far * local_transform;
        
            // Compute the MVP matrix
            let mvp_matrix = view_projection_matrix * model_matrix;
        
            // Set the MVP matrix uniform in the shader
            let transformation_location = gl::GetUniformLocation(shader_program, std::ffi::CString::new("transformationMatrix").unwrap().as_ptr());
            gl::UniformMatrix4fv(transformation_location, 1, gl::FALSE, mvp_matrix.as_ptr());
        
            // Set the Model matrix (for normal transformation) uniform in the shader
            let model_matrix_location = gl::GetUniformLocation(shader_program, std::ffi::CString::new("modelMatrix").unwrap().as_ptr());
            gl::UniformMatrix4fv(model_matrix_location, 1, gl::FALSE, model_matrix.as_ptr());
                
            // If the node is drawable, bind the VAO and issue the draw call
            if node.vao_id != 0 && node.index_count > 0 {
                gl::BindVertexArray(node.vao_id);
                gl::DrawElements(gl::TRIANGLES, node.index_count, gl::UNSIGNED_INT, std::ptr::null());
            }
        
            // Recurse into child nodes
            for &child in &node.children {
                draw_scene(&*child, view_projection_matrix, &model_matrix, shader_program);
            }
        }
        


        // Setup shader
        let simple_shader = unsafe {
            shader::ShaderBuilder::new()
                .attach_file("./shaders/simple.vert")
                .attach_file("./shaders/simple.frag")
                .link()
        };


        loop {

            let now = std::time::Instant::now();
            let elapsed = now.duration_since(first_frame_time).as_secs_f32(); // Time in seconds since the start
            let delta_time = 2.0;
            let rotation_speed_factor = 0.02;

            // Camera Movement Logic
            let forward = glm::vec3(
                camera_yaw.cos() * camera_pitch.cos(),
                camera_pitch.sin(),
                camera_yaw.sin() * camera_pitch.cos(),
            );
            let right = glm::normalize(&glm::cross(&forward, &glm::vec3(0.0, 1.0, 0.0)));

            // Animate all helicopters and their rotors
            for i in 0..5 {
                        let offset = (i as f32) * 0.3;
                        let heading = toolbox::simple_heading_animation(elapsed + offset);
                        let height_offset = 15.0 + (i as f32) * 2.0;
                
                        helicopter_root_nodes[i].position = glm::vec3(heading.x, height_offset, heading.z);
                        helicopter_root_nodes[i].rotation.y = heading.yaw;
                        helicopter_root_nodes[i].rotation.x = heading.pitch;
                        helicopter_root_nodes[i].rotation.z = heading.roll;
                
                        // Spin the rotors
                        main_rotor_nodes[i].rotation.y += delta_time * 0.1;
                        tail_rotor_nodes[i].rotation.x += delta_time * 0.1;
                    }

            // Handle resize events
            if let Ok(mut new_size) = window_size.lock() {
                if new_size.2 {
                    context.resize(glutin::dpi::PhysicalSize::new(new_size.0, new_size.1));
                    window_aspect_ratio = new_size.0 as f32 / new_size.1 as f32;
                    (*new_size).2 = false;
                    println!("Window was resized to {}x{}", new_size.0, new_size.1);
                    unsafe { gl::Viewport(0, 0, new_size.0 as i32, new_size.1 as i32); }
                }
            }

            if let Ok(keys) = pressed_keys.lock() {
            
                for key in keys.iter() {
                    match key {
                        VirtualKeyCode::A => {
                            camera_pos -= right * delta_time;  // Move left
                            println!("Flytter til venstre: {:?}", camera_pos);
                        }
                        VirtualKeyCode::D => {
                            camera_pos += right * delta_time;  // Move right
                            println!("Flytter til høyre: {:?}", camera_pos);
                        }
                        VirtualKeyCode::W => {
                            camera_pos += forward * delta_time; // Move forward
                            println!("Flytter fremover: {:?}", camera_pos);
                        }
                        VirtualKeyCode::S => {
                            camera_pos -= forward * delta_time; // Move backward
                            println!("Flytter bakover: {:?}", camera_pos);
                        }
                        VirtualKeyCode::Space => {
                            camera_pos.y += delta_time;    // Move up
                            println!("Flytter opp: {:?}", camera_pos);
                        }
                        VirtualKeyCode::LShift => {
                            camera_pos.y -= delta_time;   // Move down
                            println!("Flytter ned: {:?}", camera_pos);
                        }
                        VirtualKeyCode::Left => camera_yaw -= delta_time * rotation_speed_factor, // Rotate left
                        VirtualKeyCode::Right => camera_yaw += delta_time * rotation_speed_factor, // Rotate right
                        VirtualKeyCode::Up => camera_pitch += delta_time * rotation_speed_factor,  // Look up
                        VirtualKeyCode::Down => camera_pitch -= delta_time * rotation_speed_factor, // Look down
                        _ => {}
                    }
                }
            }
                    
                
            let camera_forward = glm::vec3(
                camera_yaw.cos() * camera_pitch.cos(),
                camera_pitch.sin(),
                camera_yaw.sin() * camera_pitch.cos(),
            );
        
            let view_matrix = glm::look_at(
                &camera_pos,
                &(camera_pos + camera_forward),
                &glm::vec3(0.0, 1.0, 0.0),
            );

        
            let view_projection_matrix = perspective_matrix * view_matrix;
            //print!("{}", view_projection_matrix);             

            unsafe {
                gl::ClearColor(0.0, 0.0, 0.0, 1.0);
                gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
        
                simple_shader.activate();
        
/*                 let gray_color = glm::vec3(0.5, 0.5, 0.5);
 */                draw_scene(&root_node, &view_projection_matrix, &glm::identity(), simple_shader.program_id);
                
    /*             let helicopter_colors: Vec<glm::Vec3> = vec![
                    glm::vec3(1.0, 0.0, 0.0),  // Red
                    glm::vec3(0.0, 1.0, 0.0),  // Green
                    glm::vec3(0.0, 0.0, 1.0),  // Blue
                    glm::vec3(1.0, 1.0, 0.0),  // Yellow
                    glm::vec3(1.0, 0.0, 1.0),  // Magenta
                ];
                
                for (i, color) in helicopter_colors.iter().enumerate() {
                    draw_scene(&helicopter_root_nodes[i], &view_projection_matrix, &glm::identity(), simple_shader.program_id, color);
                } */
            }
     
                // Display the new color buffer on the display
                context.swap_buffers().unwrap(); // we use "double buffering" to avoid artifacts
            }
    });


    // == //
    // == // From here on down there are only internals.
    // == //


    // Keep track of the health of the rendering thread
    let render_thread_healthy = Arc::new(RwLock::new(true));
    let render_thread_watchdog = Arc::clone(&render_thread_healthy);
    thread::spawn(move || {
        if !render_thread.join().is_ok() {
            if let Ok(mut health) = render_thread_watchdog.write() {
                println!("Render thread panicked!");
                *health = false;
            }
        }
    });

    // Start the event loop -- This is where window events are initially handled
    el.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        // Terminate program if render thread panics
        if let Ok(health) = render_thread_healthy.read() {
            if *health == false {
                *control_flow = ControlFlow::Exit;
            }
        }

        match event {
            Event::WindowEvent { event: WindowEvent::Resized(physical_size), .. } => {
                println!("New window size received: {}x{}", physical_size.width, physical_size.height);
                if let Ok(mut new_size) = arc_window_size.lock() {
                    *new_size = (physical_size.width, physical_size.height, true);
                }
            }
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit;
            }
            // Keep track of currently pressed keys to send to the rendering thread
            Event::WindowEvent { event: WindowEvent::KeyboardInput {
                    input: KeyboardInput { state: key_state, virtual_keycode: Some(keycode), .. }, .. }, .. } => {
                println!("Tastetrykk: {:?}, State: {:?}", keycode, key_state);
                if let Ok(mut keys) = arc_pressed_keys.lock() {
                    match key_state {
                        Released => {
                            if keys.contains(&keycode) {
                                let i = keys.iter().position(|&k| k == keycode).unwrap();
                                keys.remove(i);
                            }
                        },
                        Pressed => {
                            if !keys.contains(&keycode) {
                                keys.push(keycode);
                            }
                        }
                    }
                }

                // Handle Escape and Q keys separately
                match keycode {
                    Escape => { *control_flow = ControlFlow::Exit; }
                    Q      => { *control_flow = ControlFlow::Exit; }
                    _      => { }
                }
            }
            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
                // Accumulate mouse movement
                if let Ok(mut position) = arc_mouse_delta.lock() {
                    *position = (position.0 + delta.0 as f32, position.1 + delta.1 as f32);
                }
            }
            _ => { }
        }
    });
}