import math
import time
import sys
import socket
import queue

import dearpygui.dearpygui as dpg
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from multiprocessing.pool import ThreadPool

import wastehawk.util as util
from wastehawk.core import Wastehawk
from wastehawk.detector import Detector
from wastehawk.inputs import VideoInput

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock_thread = None

old_data = ["0", "0", "0", "0", "0", "0"]

can_log = False
log_queue = queue.Queue()

drone_positions = []

frame_thread = None
export_thread = None
thread_pool = ThreadPool(processes=3)
video_input = None

image_latency = 0.0
data_latency = 0.0
last_export_time = 0.0
start_time = 0.0

dl_t = 0
il_t = 0

# Retreive data from localhost.
def fetch_drone_data():
    global thread_pool
    global sock_thread
    global sock
    global old_data

    global data_latency
    global dl_t

    data = None
    if sock_thread != None and sock_thread.ready():
        data_latency = time.time() - dl_t
        data, addr = sock_thread.get()
        data = data.decode().split(" ")
        sock_thread = None

    if sock_thread == None:
        dl_t = time.time()
        sock_thread = thread_pool.apply_async(lambda: sock.recvfrom(1240), ())

    if data != None:
        old_data = data
        return data
    
    return old_data

# For intercepting print calls
class GUIStdOut:
    def __init__(self, stdout, callback):
        self.stdout = stdout
        self.callback = callback

    def write(self, string):
        outstr = string
        if string != '\n':
            outstr = (f"[Wastehawk {time.strftime('%H:%M:%S')}] ") + outstr
            self.callback(outstr)

        self.stdout.write(outstr)

    def flush(self):
        self.stdout.flush()

def _split_xy(array):
    return [float(x[0]) for x in array], [float(y[1]) for y in array]

def _export_data(file: str):
    global wastehawk_core
    global drone_positions

    path_x, path_y = _split_xy(drone_positions.copy())
    path_z = [float(z[2]) for z in drone_positions]
    path_time = [float(t[3]) for t in drone_positions]


    trash_x, trash_y = _split_xy(wastehawk_core.trash_positions.copy())

    path_data = pd.DataFrame({'time': path_time, 'x': path_x, 'y': path_y, 'z': path_z})
    path_data = path_data.drop_duplicates(subset=['x', 'y', 'z'])
    trash_data = pd.DataFrame({'x': trash_x, 'y': trash_y})

    trash_data.to_csv(str(file + '_trash_data.csv'))
    path_data.to_csv(str(file + '_path_data.csv'))

    fig, (trash_heatmap, trash_plot, path_plot) = plt.subplots(1, 3)
    fig.set_size_inches((18, 6))
    trash_heatmap.set_aspect('equal', adjustable='box')
    trash_plot.set_aspect('equal', adjustable='box')
    path_plot.set_aspect('equal', adjustable='box')

    sns.kdeplot(data=trash_data, x='x', y='y', fill=True, thresh=0, levels=10, ax=trash_heatmap, cmap='rocket')
    sns.scatterplot(data=trash_data, x='x', y='y', ax=trash_plot)
    sns.lineplot(data=path_data, x='x', y='y', sort=False, ax=path_plot)

    plt.savefig(str(file + '_graphs.png'), dpi=75)

def export(file: str = 'output'):
    global export_thread
    global last_export_time

    export_thread = thread_pool.apply_async(_export_data, [file])
    last_export_time = time.time()

def export_as(sender, data):
    filename = data['file_path_name']
    export(filename)

def print_callback(message):
    global can_log
    global log_queue

    if can_log:
        dpg.add_text(message, parent='log panel')
        dpg.set_y_scroll('log panel', -1.0)
    else:
        log_queue.put(message)

def copy_logs():
    logs = dpg.get_item_children('log panel')[1]
    outstr = ''
    for log in logs:
        outstr += (dpg.get_value(log) + "\n")

    dpg.set_clipboard_text(outstr)
    print(f"Copied {len(logs)} lines to clipboard.")

def _scale_and_fit_view():
    view_width, view_height = dpg.get_item_rect_size("subpanel 2")

    ratio = (1920 / 1080, 1080 / 1920)

    frame_width = view_width
    frame_height = view_width * ratio[1]

    frame_width = min(frame_width, 1920)
    frame_height = min(frame_height, 1080)

    dpg.configure_item("source_view", pmax=(frame_width, frame_height))
    dpg.configure_item("source_display", width=view_width, height=view_height)

    transform = [(view_width / 2) - (frame_width / 2), (view_height / 2) - (frame_height / 2)]
    mat = dpg.create_translation_matrix(transform)

    dpg.apply_transform('source_layer', mat)
    dpg.apply_transform('boxes_layer', mat)

def _draw_detection_boxes(shape, boxes):
    dpg.delete_item('boxes_layer')

    with dpg.draw_node(tag='boxes_layer', parent='source_display'):

        for box in boxes:
            x1 = (box[0] / shape[1]) * dpg.get_item_configuration('source_view')['pmax'][0]
            x2 = (box[2] / shape[1]) * dpg.get_item_configuration('source_view')['pmax'][0]
            y1 = (box[1] / shape[0]) * dpg.get_item_configuration('source_view')['pmax'][1]
            y2 = (box[3] / shape[0]) * dpg.get_item_configuration('source_view')['pmax'][1]

            dpg.draw_text([x1, y1 - 17], text=f'{box[4]:.2f}', size=15)
            dpg.draw_polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]], color=(0, 125, 255, 255), thickness=2)

            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2

            dpg.draw_line([mx - 10, my],  [mx + 10, my], color=(255, 0, 0, 255), thickness=2)
            dpg.draw_line([mx, my - 10],  [mx, my + 10], color=(255, 0, 0, 255), thickness=2)

def change_config_callback():
    denoise_resolution = dpg.get_value("config denoise resolution")
    minimum_confidence = dpg.get_value("config minimum confidence")
    
    h_fov = dpg.get_value("config hfov")
    v_fov = dpg.get_value("config vfov")

    global wastehawk_core
    wastehawk_core.apply_configs(h_fov, v_fov, denoise_resolution)

def tooltip(text: str):
    with dpg.tooltip(parent=dpg.last_item()):
        dpg.add_text(text)

def display_label(tag, text, value):
    with dpg.table_row():
        dpg.add_text(text)
        dpg.add_text(tag=tag, default_value=value)

def build_side_panel():
    with dpg.table(header_row=False):
        dpg.add_table_column()
        dpg.add_table_column()

        display_label('drone x label', 'Drone X', 0)
        tooltip("Offset from drone origin point")
        display_label('drone y label', 'Drone Y', 0)
        tooltip("Offset from drone origin point")
        display_label('drone rot label', 'Drone Rot.', 0)
        display_label('drone alt label', 'Drone Alt.', 0)

    dpg.add_spacer(height=15)

    with dpg.table(header_row=False):
        dpg.add_table_column()
        dpg.add_table_column()

        display_label('detection latency label', 'Inference Lag', '0ms')
        tooltip("Time it takes for inference to finish")
        display_label('denoise latency label', 'Denoise Lag', '0ms')
        tooltip("Time it takes to denoise results")
        display_label('data latency label', 'Data Latency', '0ms')
        tooltip("Time it takes to retreive drone position & orientation data")
        display_label('image latency label', 'Image Latency', '0ms')
        tooltip("Time it takes to retreive drone footage")
    
    dpg.add_spacer(height=15)

    with dpg.collapsing_header(label="Projection Settings"):
        dpg.add_input_float(default_value=46.545, width=185, callback=change_config_callback, tag='config hfov')
        tooltip('Horizontal FOV')
        dpg.add_input_float(default_value=30.936, width=185, callback=change_config_callback, tag='config vfov')
        tooltip('Vertical FOV')

    with dpg.collapsing_header(label="Output Filtering"):
        dpg.add_input_float(default_value=0.1, width=185, callback=change_config_callback, tag='config denoise resolution')
        tooltip('Denoise Resolution')
        dpg.add_input_float(default_value=0.45, callback=change_config_callback, width=185)
        tooltip('Minimum Confidence')

def build_source_panel():
    with dpg.drawlist(tag="source_display", width=1920, height=1080) as display:

        with dpg.draw_node(tag='source_layer'):
            dpg.draw_image('source_frame', (0, 0), (1920, 1080), tag='source_view', uv_min=(0, 0), uv_max=(1, 1))
                
        dpg.add_draw_node(tag='boxes_layer', parent=display)

def build_position_plot():
    with dpg.plot(tag='position plot', no_title=True, no_menus=True, no_mouse_pos=True, width=-1, height=-1, equal_aspects=True) as plot:
        dpg.add_plot_axis(dpg.mvXAxis, label="x", tag="position_x_axis")
        dpg.add_plot_axis(dpg.mvYAxis, label="y", tag="position_y_axis")

        axis = dpg.last_item()

        dpg.add_line_series([], [], parent=axis, tag='drone_position_series')
        dpg.add_scatter_series([], [], parent=axis, tag='trash_position_series')

        with dpg.draw_node(tag='drone_heading_overlay', parent=plot) as node:
            dpg.draw_circle((0, 0), 0.075, fill=(255, 255, 0), parent=node)
            dpg.draw_arrow((0, 1), (0, 0), color=(255, 255, 0), thickness=0.05, parent=node, size=0.05)

def build_menu_bar():
    with dpg.menu(label="File"):
        dpg.add_menu_item(label="Save", callback=export)
        tooltip("Save detection results as file")
        dpg.add_menu_item(label="Save As", callback=lambda: dpg.show_item('export_file_dialog'))
        tooltip("Save detection results as file at a location")
    
    with dpg.menu(label="Help"):
        dpg.add_menu_item(label="Manual")
        tooltip("How to use Wastehawk")
        dpg.add_menu_item(label="Source Code")
        tooltip("Wastehawk's source code")
        dpg.add_menu_item(label="Paper")
        tooltip("Wastehawk extended abstract")

    
def build_view_panels():
    with dpg.table(header_row=False):
        dpg.add_table_column()
        dpg.add_table_column()

        with dpg.table_row():
            with dpg.child_window(tag='subpanel 1'):
                build_position_plot()                            
            with dpg.child_window(tag='subpanel 2', no_scrollbar=True, no_scroll_with_mouse=True):
                build_source_panel()

def build_gui():
    dpg.create_context()

    with dpg.texture_registry():
        dpg.add_raw_texture(tag='source_frame', width=1920, height=1080, default_value=np.zeros((1920 * 1080) * 3), format=dpg.mvFormat_Float_rgb)

    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (59, 59, 59), category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8, category=dpg.mvThemeCat_Core)
    dpg.bind_theme(global_theme)

    dpg.add_file_dialog(show=False, callback=export_as, tag='export_file_dialog')

    with dpg.window(tag="root", no_scrollbar=True):
        with dpg.menu_bar():
            build_menu_bar()

        with dpg.table(header_row=False):
            dpg.add_table_column(width_fixed=True, width=200)
            dpg.add_table_column(width_fixed=False, width=200)

            with dpg.table_row():
                with dpg.child_window(tag='panel 1', width=200, height=520):
                    build_side_panel()
                
                with dpg.child_window(tag='panel 2', height=520, no_scrollbar=True, autosize_x=True):
                    build_view_panels()
        
        with dpg.child_window(tag='panel 3', height=150):
            with dpg.group(horizontal=True):
                dpg.add_button(label='Clear', callback=lambda: dpg.delete_item('log panel', children_only=True))
                dpg.add_button(label='Copy', callback=copy_logs)
            dpg.add_child_window(tag='log panel', autosize_x=True, autosize_y=True)

        with dpg.group(horizontal=True, horizontal_spacing=15):
            dpg.add_text(tag="fps label", default_value="FPS")
            dpg.add_text("Wastehawk 1.0 - DEMO GUI")
        

    dpg.set_primary_window('root', True)
    dpg.create_viewport(title="Wastehawk Demo", width=1080, height=720, clear_color=(59, 59, 59))
    dpg.setup_dearpygui()
    dpg.show_viewport()

if __name__ == '__main__':
    sys.stdout = GUIStdOut(sys.stdout, print_callback)

    wastehawk_core = Wastehawk().set_detector(Detector()).apply_configs(46.545, 30.936)
    video_input = VideoInput(1)

    build_gui()

    can_log = True
    start_time = time.time()

    while dpg.is_dearpygui_running():
            dpg.set_value('fps label', f'FPS: {dpg.get_frame_rate()}')

            # Resize Panels
            vp_h = dpg.get_viewport_height()
            vp_w = dpg.get_viewport_width()

            dpg.set_item_height('panel 1', vp_h - 260)
            dpg.set_item_height('panel 2', vp_h - 260)

            # Pass drone data to Wastehawk
            data = fetch_drone_data()
            data = [float(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4])]

            ret, frame = video_input.read()
            wastehawk_core.update_drone_data(data[0], data[1], data[2], data[3])
            wastehawk_core.update_detection(frame)

            if wastehawk_core.output:
                _draw_detection_boxes(wastehawk_core.output[0], wastehawk_core.output[1])
            
            _scale_and_fit_view() # Resize view

            trash_positions = wastehawk_core.trash_positions

            # Indicators and plot
            dpg.set_value('detection latency label', f'{(wastehawk_core.latency):.3f} ms')
            dpg.set_value('denoise latency label', f'{(wastehawk_core.denoise_latency):.3f} ms')
            dpg.set_value('data latency label', f'{(data_latency):.3f} ms')
            dpg.set_value('image latency label', f'{(image_latency):.3f} ms')

            dpg.set_value('drone x label', f'{data[0]:.5f}')
            dpg.set_value('drone y label', f'{data[1]:.5f}')
            dpg.set_value('drone alt label', f'{data[2]:.5f}')
            dpg.set_value('drone rot label', f'{data[3]:.5f}')
            drone_positions.append([data[0], data[1], data[2], time.time() - start_time])
            dpg.set_value('drone_position_series', [[p[0] for p in drone_positions], [p[1] for p in drone_positions]])
            dpg.set_value('trash_position_series', [[q[0] for q in trash_positions], [q[1] for q in trash_positions]])

            trans_mat = dpg.create_translation_matrix([data[0], data[1]])
            rot_mat = dpg.create_rotation_matrix(math.pi*(-data[3])/180.0, (0, 0, 1))
            dpg.apply_transform('drone_heading_overlay', trans_mat * rot_mat)

            # Handle logger
            while not log_queue.empty():
                print_callback(log_queue.get(block=False))

            # Handle exporting
            if export_thread != None and export_thread.ready():
                print(f"File export finished. ({(time.time() - last_export_time):.3f}ms)")
                export_thread = None

            # Change display frame
            if frame.any():
                if frame_thread != None and frame_thread.ready():
                    image_latency = time.time() - il_t
                    dpg.set_value('source_frame', frame_thread.get())
                    frame_thread = None
                
                # The conversion function is very laggy, so we use a thread.
                if frame_thread == None:
                   il_t = time.time()
                   frame_thread = thread_pool.apply_async(lambda image: util.cv2_to_dpg(image, (1920, 1080)), [frame])

            dpg.render_dearpygui_frame()

    dpg.destroy_context()

    sys.stdout = sys.stdout.stdout
