###############################################
# MVC Project 1: 3D Vector Function with Components
# Incorporates Interactive Plot with Frenet-Serret Frame and Component Graphs
# Author: Mia Jaenike
# Last Updated: October 13, 2025
################################################

# Necessary imports
import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------- helpers -------------------------
def _safe_eval(expr, t_vals) -> np.ndarray:
    '''
    Evaluate a mathematical expression with given variables (safely)
    Params:
        expr: string expression in terms of 't' and numpy functions
        t_vals: array-like of t values
    Returns:
        evaluated array
    '''
    ns = {
        'np': np,
        't': t_vals,
        'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
        'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan,
        'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
        'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt, 'abs': np.abs,
        'pi': np.pi, 'e': np.e
    }
    return eval(expr, {'__builtins__': {}}, ns)

def _range_with_pad(values, pad_frac=0.15) -> list[float]:
    '''
    Compute range with padding
    Params:
        values: array-like of numeric values
        pad_frac: fraction of span to pad on each side
    Returns:
        [min - pad, max + pad]
    '''
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        span = max(abs(hi) if np.isfinite(hi) else 1.0, 1.0) # avoid zero span
    pad = pad_frac * span
    return [lo - pad, hi + pad]

def _unit_rows(vectors, eps=1e-12) -> tuple[np.ndarray, np.ndarray]:
    '''
    Normalize rows of a 2D array, returns unit vectors and their norms
    Params:
        vectors: (N, 3) array of N vectors
        eps: small value to avoid division by zero
    Returns:
        u: (N, 3) array of unit vectors
        norms: (N,) array of original vector norms
    '''
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms_safe = np.where(norms < eps, 1.0, norms)
    u = vectors / norms_safe
    u[norms.flatten() < eps] = 0.0
    return u, norms.flatten()

def _build_tail_coords(x_full, y_full, z_full, idx, tail_len) -> tuple[list[float], list[float], list[float]]:
    '''
    Build tail coordinates for the curve up to index idx
    Params:
        x_full, y_full, z_full: full coordinate arrays
        idx: current index
        tail_len: length of the tail
    Returns:
        xs, ys, zs: lists of coordinates for the tail with None separators
    '''
    start = max(1, idx - tail_len + 1)
    xs, ys, zs = [], [], []
    for k in range(start, idx + 1):
        xs += [x_full[k - 1], x_full[k], None]
        ys += [y_full[k - 1], y_full[k], None]
        zs += [z_full[k - 1], z_full[k], None]
    return xs, ys, zs

# --------------------- main plotting ----------------------
def plot_vector_with_components(functions, t_min=0, t_max=10, num_points=600,
                                responsive=False, guides=False,
                                normalize=False, pad_frac=0.15,
                                save_html_path=None, show_plane=False,
                                arrow_scale=0.08,
                                # readout controls
                                readout_mode='global',          # 'global' or 'panel'
                                readout_x=0.02, readout_y=0.6, # overlay coords (0..1 inside the band)
                                readout_band=0.015,
                                readout_gap=0.03,
                                readout_where='top',
                                readout_xp=0.08, readout_yp=0.92, # panel coords (x(t) panel)
                                readout_size=12) -> None:
    '''
    Plot 3D vector function with its components and Frenet-Serret frame
    Params:
        functions: list of 3 strings for x(t), y(t), z(t)
        t_min, t_max: parameter range
        num_points: number of points to sample
        responsive: whether plot is responsive (essentially, resizable)
        guides: whether to show guide lines in component plots
        normalize: whether to normalize component plots
        pad_frac: padding fraction for axes
        save_html_path: if given, save plot to this HTML file
        show_plane: whether to show osculating plane
        arrow_scale: scale for Frenet-Serret arrows
        readout_mode: 'global' or 'panel' for speed/curvature/tors
        readout_x, readout_y: overlay coords for global readout
        readout_band: height of the readout band (fraction)
        readout_gap: gap between readout band and main plot (fraction)
        readout_where: 'top' or 'bottom' for global readout band position
        readout_xp, readout_yp: panel coords for panel readout
        readout_size: font size for readout text
    Returns:
        None
    '''
    # parameter and eval
    t_full = np.linspace(t_min, t_max, num_points)
    x_full = _safe_eval(functions[0], t_full)
    y_full = _safe_eval(functions[1], t_full)
    z_full = _safe_eval(functions[2], t_full)

    # derivatives (central differences, meaning uneven spacing not supported)
    dx = np.gradient(x_full, t_full)
    dy = np.gradient(y_full, t_full)
    dz = np.gradient(z_full, t_full)
    d2x = np.gradient(dx, t_full)
    d2y = np.gradient(dy, t_full)
    d2z = np.gradient(dz, t_full)
    d3x = np.gradient(d2x, t_full)
    d3y = np.gradient(d2y, t_full)
    d3z = np.gradient(d2z, t_full)

    r1 = np.stack([dx, dy, dz], axis=1)   # r'(t)
    r2 = np.stack([d2x, d2y, d2z], axis=1)  # r''(t)
    r3 = np.stack([d3x, d3y, d3z], axis=1)  # r'''(t)

    T, speed = _unit_rows(r1)             # unit tangent and speed
    cross12 = np.cross(r1, r2)            # r' x r''
    B, cross12_norm = _unit_rows(cross12) # unit binormal
    N_temp = np.cross(B, T)              
    N, _ = _unit_rows(N_temp)

    # curvature and torsion
    with np.errstate(divide='ignore', invalid='ignore'):
        curvature = np.where(speed > 1e-12, np.linalg.norm(cross12, axis=1) / (speed ** 3 + 1e-18), 0.0)
        triple = (r1[:, 0] * (r2[:, 1] * r3[:, 2] - r2[:, 2] * r3[:, 1]) -
                  r1[:, 1] * (r2[:, 0] * r3[:, 2] - r2[:, 2] * r3[:, 0]) +
                  r1[:, 2] * (r2[:, 0] * r3[:, 1] - r2[:, 1] * r3[:, 0]))
        torsion = np.where(cross12_norm > 1e-12, triple / (cross12_norm ** 2 + 1e-18), 0.0)

    # right-panel components
    if normalize:
        def norm(v):
            m = np.nanmax(np.abs(v))
            m = m if m > 1e-12 else 1.0 # avoid division by zero
            return v / m
        x_comp, y_comp, z_comp = norm(x_full), norm(y_full), norm(z_full)
        ylab_x, ylab_y, ylab_z = "x(t) (normalized)", "y(t) (normalized)", "z(t) (normalized)"
    else:
        x_comp, y_comp, z_comp = x_full, y_full, z_full
        ylab_x, ylab_y, ylab_z = "x(t)", "y(t)", "z(t)"

    # axis ranges
    t_range = [float(t_min), float(t_max)]
    x_range3d = _range_with_pad(x_full, pad_frac)
    y_range3d = _range_with_pad(y_full, pad_frac)
    z_range3d = _range_with_pad(z_full, pad_frac)
    xr = _range_with_pad(x_comp, pad_frac)
    yr = _range_with_pad(y_comp, pad_frac)
    zr = _range_with_pad(z_comp, pad_frac)

    # animation params
    num_frames = 120
    tail_length = max(20, num_points // 20)

    # subplots
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{'type': 'scene', 'rowspan': 3}, {'type': 'xy'}],
            [None, {'type': 'xy'}],
            [None, {'type': 'xy'}]],
        row_heights=[0.34, 0.33, 0.33],
        column_widths=[0.64, 0.36],
        horizontal_spacing=0.06,
        vertical_spacing=0.06,
        subplot_titles=(None, 'x(t) vs t', 'y(t) vs t', 'z(t) vs t')
    )

    # 3D curve traces
    fig.add_trace(go.Scatter3d(x=x_full, y=y_full, z=z_full, mode='lines',
                            line=dict(width=3), opacity=0.3, showlegend=False, hoverinfo='skip'), row=1, col=1)  # 0
    fig.add_trace(go.Scatter3d(mode='lines', line=dict(width=3), opacity=0.45, showlegend=False, hoverinfo='skip'), row=1, col=1)  # 1
    fig.add_trace(go.Scatter3d(mode='lines', line=dict(width=4), opacity=0.9, showlegend=False, hoverinfo='skip'), row=1, col=1)   # 2
    fig.add_trace(go.Scatter3d(mode='markers',
                            marker=dict(size=8, color='black', line=dict(width=2, color='black')),
                            showlegend=False, hoverinfo='text', text=[]), row=1, col=1)  # 3

    # right panels (lines and moving dots)
    fig.add_trace(go.Scatter(x=t_full, y=x_comp, mode='lines', showlegend=False), row=1, col=2)  # 4
    fig.add_trace(go.Scatter(x=[t_full[0]], y=[x_comp[0]], mode='markers',
                            marker=dict(size=9, color='black'), showlegend=False, hoverinfo='skip'), row=1, col=2)  # 5
    fig.add_trace(go.Scatter(x=t_full, y=y_comp, mode='lines', showlegend=False), row=2, col=2)  # 6
    fig.add_trace(go.Scatter(x=[t_full[0]], y=[y_comp[0]], mode='markers',
                            marker=dict(size=9, color='black'), showlegend=False, hoverinfo='skip'), row=2, col=2)  # 7
    fig.add_trace(go.Scatter(x=t_full, y=z_comp, mode='lines', showlegend=False), row=3, col=2)  # 8
    fig.add_trace(go.Scatter(x=[t_full[0]], y=[z_comp[0]], mode='markers',
                            marker=dict(size=9, color='black'), showlegend=False, hoverinfo='skip'), row=3, col=2)  # 9

    # Frenet-Serret (FS) colors
    fs_color_T = '#2ca02c'  # green
    fs_color_N = '#d62728'  # red
    fs_color_B = '#1f77b4'  # blue

    # initialize FS frame at t=t_min
    # use a robust base length so one huge axis doesn't blow up the cones
    # fixed absolute arrow length (independent of curve scale)
    arrow_len = 1.0
    cone_size = 0.35 * arrow_len

    idx0 = 0
    px0, py0, pz0 = x_full[idx0], y_full[idx0], z_full[idx0]
    Tx0, Ty0, Tz0 = T[idx0]
    Nx0, Ny0, Nz0 = N[idx0]
    Bx0, By0, Bz0 = B[idx0]
    T_end0 = (px0 + arrow_len * Tx0, py0 + arrow_len * Ty0, pz0 + arrow_len * Tz0)
    N_end0 = (px0 + arrow_len * Nx0, py0 + arrow_len * Ny0, pz0 + arrow_len * Nz0)
    B_end0 = (px0 + arrow_len * Bx0, py0 + arrow_len * By0, pz0 + arrow_len * Bz0)

    # FS lines & cones
    fig.add_trace(go.Scatter3d(x=[px0, T_end0[0]], y=[py0, T_end0[1]], z=[pz0, T_end0[2]],
                            mode='lines', line=dict(width=6, color=fs_color_T),
                            showlegend=False, hoverinfo='skip'), row=1, col=1)  # 10
    fig.add_trace(go.Scatter3d(x=[px0, N_end0[0]], y=[py0, N_end0[1]], z=[pz0, N_end0[2]],
                            mode='lines', line=dict(width=6, color=fs_color_N),
                            showlegend=False, hoverinfo='skip'), row=1, col=1)  # 11
    fig.add_trace(go.Scatter3d(x=[px0, B_end0[0]], y=[py0, B_end0[1]], z=[pz0, B_end0[2]],
                            mode='lines', line=dict(width=6, color=fs_color_B),
                            showlegend=False, hoverinfo='skip'), row=1, col=1)  # 12
    fig.add_trace(go.Cone(x=[T_end0[0]], y=[T_end0[1]], z=[T_end0[2]], u=[Tx0], v=[Ty0], w=[Tz0],
                        sizemode='absolute', sizeref=cone_size, anchor='tip',
                        showscale=False, colorscale=[[0, fs_color_T], [1, fs_color_T]]), row=1, col=1)  # 13
    fig.add_trace(go.Cone(x=[N_end0[0]], y=[N_end0[1]], z=[N_end0[2]], u=[Nx0], v=[Ny0], w=[Nz0],
                        sizemode='absolute', sizeref=cone_size, anchor='tip',
                        showscale=False, colorscale=[[0, fs_color_N], [1, fs_color_N]]), row=1, col=1)  # 14
    fig.add_trace(go.Cone(x=[B_end0[0]], y=[B_end0[1]], z=[B_end0[2]], u=[Bx0], v=[By0], w=[Bz0],
                        sizemode='absolute', sizeref=cone_size, anchor='tip',
                        showscale=False, colorscale=[[0, fs_color_B], [1, fs_color_B]]), row=1, col=1)  # 15

    # osculating plane placeholder
    fig.add_trace(go.Mesh3d(x=[], y=[], z=[], i=[0, 1, 2, 0, 2, 3], j=[1, 2, 3, 1, 3, 0], k=[2, 3, 0, 3, 0, 1],
                            color='gray', opacity=0.15, showscale=False, hoverinfo='skip', showlegend=False), row=1, col=1)  # 16

    # layout and axes
    fig.update_layout(template='plotly_white',
        scene=dict(
            xaxis=dict(title='x', range=x_range3d),
            yaxis=dict(title='y', range=y_range3d),
            zaxis=dict(title='z', range=z_range3d),
            aspectmode='cube'
        ),
        xaxis=dict(title='t', range=t_range, fixedrange=True, uirevision="stay"),
        yaxis=dict(title=ylab_x, range=xr, fixedrange=True, uirevision="stay"),
        xaxis2=dict(title='t', range=t_range, fixedrange=True, uirevision="stay"),
        yaxis2=dict(title=ylab_y, range=yr, fixedrange=True, uirevision="stay"),
        xaxis3=dict(title='t', range=t_range, fixedrange=True, uirevision="stay"),
        yaxis3=dict(title=ylab_z, range=zr, fixedrange=True, uirevision="stay"),
        width=1200, height=820,
        title=dict(text=f"r(t) = ⟨{functions[0]}, {functions[1]}, {functions[2]}⟩ & components", x=0.5),
        margin=dict(l=40, r=20, t=70, b=96),
        uirevision="stay"
    )

    # add overlsy axes covering the whole figure top band for GLOBAL readout
    if str(readout_where).lower().startswith('b'):
        y4_domain = [0.0, max(0.001, float(readout_band))]
        slider_y = 0.06  # lift slider above the band
    else:
        y4_domain = [max(0.0, 1.0 - float(readout_gap) - max(0.001, float(readout_band))), max(0.0, 1.0 - float(readout_gap))]
        slider_y = 0.02
    fig.update_layout(
        xaxis4=dict(domain=[0.0, 1.0], visible=False, range=[0,1]),
        yaxis4=dict(domain=y4_domain, visible=False, range=[0,1])
    )

    readout_trace_index = None
    if readout_mode == 'panel':
        # position inside the x(t) panel
        readout_xp = max(0.0, min(1.0, float(readout_xp)))
        readout_yp = max(0.0, min(1.0, float(readout_yp)))
        rx = t_range[0] + readout_xp * (t_range[1] - t_range[0])
        ry = yr[0] + readout_yp * (yr[1] - yr[0])
        fig.add_trace(go.Scatter(x=[rx], y=[ry], text=["speed: -- | κ: -- | τ: --"],
                                mode='text', showlegend=False, textposition='middle left',
                                textfont=dict(size=readout_size)), row=1, col=2)
        readout_trace_index = len(fig.data) - 1
    else:
        # global: use overlay axes text trace (no annotations to avoid ghosting)
        gx = max(0.0, min(1.0, float(readout_x)))
        gy = max(0.0, min(1.0, float(readout_y)))
        fig.add_trace(go.Scatter(x=[gx], y=[gy], text=[f"speed: {speed[0]:.3f} | κ: {curvature[0]:.3f} | τ: {torsion[0]:.3f}"],
                                mode='text', showlegend=False, textposition='middle left',
                                textfont=dict(size=readout_size)))
        # attach to overlay axes
        fig.data[-1].update(xaxis='x4', yaxis='y4')
        readout_trace_index = len(fig.data) - 1

    # build frames
    frames = []
    steps = []

    x_lo, x_hi = float(np.min(x_comp)), float(np.max(x_comp))
    y_lo, y_hi = float(np.min(y_comp)), float(np.max(y_comp))
    z_lo, z_hi = float(np.min(z_comp)), float(np.max(z_comp))

    for i in range(num_frames + 1):
        idx = int(i / num_frames * (num_points - 1))

        traced_x = x_full[:max(0, idx - tail_length)]
        traced_y = y_full[:max(0, idx - tail_length)]
        traced_z = z_full[:max(0, idx - tail_length)]

        tail_x, tail_y, tail_z = _build_tail_coords(x_full, y_full, z_full, idx, tail_length)

        px, py, pz = x_full[idx], y_full[idx], z_full[idx]
        current_x = [px]; current_y = [py]; current_z = [pz]
        hover_text = [f"t = {t_full[idx]:.3f}<br>x={px:.3f}<br>y={py:.3f}<br>z={pz:.3f}"]

        dot_x = [t_full[idx]], [x_comp[idx]]
        dot_y = [t_full[idx]], [y_comp[idx]]
        dot_z = [t_full[idx]], [z_comp[idx]]

        Tx, Ty, Tz = T[idx]; Nx, Ny, Nz = N[idx]; Bx, By, Bz = B[idx]
        T_end = (px + arrow_len * Tx, py + arrow_len * Ty, pz + arrow_len * Tz)
        N_end = (px + arrow_len * Nx, py + arrow_len * Ny, pz + arrow_len * Nz)
        B_end = (px + arrow_len * Bx, py + arrow_len * By, pz + arrow_len * Bz)

        sp = speed[idx]; kap = curvature[idx]; tor = torsion[idx]
        readout_text = f"speed: {sp:.3f} | κ: {kap:.3f} | τ: {tor:.3f}"

        shapes = []
        if guides:
            shapes = [
                dict(type='line', x0=t_full[idx], x1=t_full[idx], y0=x_lo, y1=x_hi, xref='x', yref='y'),
                dict(type='line', x0=t_full[idx], x1=t_full[idx], y0=y_lo, y1=y_hi, xref='x2', yref='y2'),
                dict(type='line', x0=t_full[idx], x1=t_full[idx], y0=z_lo, y1=z_hi, xref='x3', yref='y3'),
            ]

        frame_data = [
            go.Scatter3d(x=traced_x, y=traced_y, z=traced_z),         # 1
            go.Scatter3d(x=tail_x, y=tail_y, z=tail_z),               # 2
            go.Scatter3d(x=current_x, y=current_y, z=current_z, text=hover_text,
                        marker=dict(size=8, color='black', line=dict(width=2, color='black'))),  # 3
            go.Scatter(x=dot_x[0], y=dot_x[1], marker=dict(size=9, color='black')),  # 5
            go.Scatter(x=dot_y[0], y=dot_y[1], marker=dict(size=9, color='black')),  # 7
            go.Scatter(x=dot_z[0], y=dot_z[1], marker=dict(size=9, color='black')),  # 9
            go.Scatter3d(x=[px, T_end[0]], y=[py, T_end[1]], z=[pz, T_end[2]], line=dict(width=6, color=fs_color_T)),  # 10
            go.Scatter3d(x=[px, N_end[0]], y=[py, N_end[1]], z=[pz, N_end[2]], line=dict(width=6, color=fs_color_N)),  # 11
            go.Scatter3d(x=[px, B_end[0]], y=[py, B_end[1]], z=[pz, B_end[2]], line=dict(width=6, color=fs_color_B)),  # 12
            go.Cone(x=[T_end[0]], y=[T_end[1]], z=[T_end[2]], u=[Tx], v=[Ty], w=[Tz],
                    sizemode='absolute', sizeref=cone_size, anchor='tip',
                    showscale=False, colorscale=[[0, fs_color_T], [1, fs_color_T]]),  # 13
            go.Cone(x=[N_end[0]], y=[N_end[1]], z=[N_end[2]], u=[Nx], v=[Ny], w=[Nz],
                    sizemode='absolute', sizeref=cone_size, anchor='tip',
                    showscale=False, colorscale=[[0, fs_color_N], [1, fs_color_N]]),  # 14
            go.Cone(x=[B_end[0]], y=[B_end[1]], z=[B_end[2]], u=[Bx], v=[By], w=[Bz],
                    sizemode='absolute', sizeref=cone_size, anchor='tip',
                    showscale=False, colorscale=[[0, fs_color_B], [1, fs_color_B]]),  # 15
            go.Mesh3d(x=[], y=[], z=[]),  # 16 (maybe replaced)
        ]

        if readout_trace_index is not None:
            # placeholder updated text for the readout trace
            rd = go.Scatter(x=[readout_x], y=[readout_y], text=[readout_text],
                            mode='text', textposition='middle left', textfont=dict(size=readout_size))
            if readout_mode == 'global':
                rd.update(xaxis='x4', yaxis='y4')
            else:
                # in panel mode use panel coords converted to absolute already used only at init
                pass
            frame_data.append(rd)

        if show_plane:
            plane_half = arrow_len * 1.5
            v1 = np.array([Tx, Ty, Tz])
            v2 = np.array([Nx, Ny, Nz])
            p_center = np.array([px, py, pz])
            corners = np.array([
                p_center - plane_half * v1 - plane_half * v2,
                p_center + plane_half * v1 - plane_half * v2,
                p_center + plane_half * v1 + plane_half * v2,
                p_center - plane_half * v1 + plane_half * v2
            ])
            plane_x, plane_y, plane_z = corners[:, 0], corners[:, 1], corners[:, 2]
            frame_data[12] = go.Mesh3d(x=plane_x, y=plane_y, z=plane_z,
                                    i=[0, 1, 2, 0, 2, 3], j=[1, 2, 3, 1, 3, 0], k=[2, 3, 0, 3, 0, 1],
                                    color='gray', opacity=0.15, showscale=False, hoverinfo='skip')

        traces_list = [1, 2, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16]
        if readout_trace_index is not None:
            traces_list.append(readout_trace_index)

        frames.append(go.Frame(
            name=str(i),
            data=frame_data,
            traces=traces_list
        ))

        steps.append(dict(
            method='animate',
            args=[[str(i)], dict(frame=dict(duration=40, redraw=True),
                                mode='immediate',
                                transition=dict(duration=0))],
            label=f"{t_full[idx]:.2f}"
        ))

    fig.frames = frames

    sliders = [dict(active=0, steps=steps, x=0.12, y=slider_y, len=0.76,
                    currentvalue=dict(visible=True, prefix='t = '),
                    pad=dict(t=50, b=10))]

    updatemenus = [dict(type='buttons', showactive=False, x=0.02, y=0.02,
                        buttons=[
                            dict(label='▶ Play', method='animate',
                                args=[None, dict(frame=dict(duration=40, redraw=True),
                                                fromcurrent=True, mode='immediate',
                                                transition=dict(duration=0))]),
                            dict(label='⏸ Pause', method='animate',
                                args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                    mode='immediate',
                                                    transition=dict(duration=0))])
                        ])]

    fig.update_layout(sliders=sliders, updatemenus=updatemenus)
    fig.update_layout(scene_camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)))

    if save_html_path:
        fig.write_html(save_html_path, include_plotlyjs='cdn', full_html=True)

    fig.show(config={'responsive': bool(responsive)})

# ---------------------- CLI wrapper ------------------------
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python [filename].py 'x(t)' 'y(t)' 'z(t)' [t_min] [t_max] "
            "[--responsive] [--guides] [--normalize] [--pad <float>] [--save-html <path>] "
            "[--plane] [--arrow <scale>] "
            "[--readout-mode global|panel] "
            "[--readout-x <0..1>] [--readout-y <0..1>] [--readout-size <int>] "
            "[--readout-xp <0..1>] [--readout-yp <0..1>]")
        sys.exit(1)

    # Parse arguments
    funcs = sys.argv[1:4]
    t_min = 0.0
    t_max = 10.0
    responsive = False
    guides = False
    normalize = False
    pad_frac = 0.15
    save_html_path = None
    show_plane = False
    arrow_scale = 0.08
    readout_mode = 'global'
    readout_x = 0.02
    readout_y = 0.975
    readout_size = 12
    readout_xp = 0.08
    readout_yp = 0.92
    readout_band = 0.015
    readout_gap = 0.03
    readout_where = 'bottom'

    # Advanced arg parsing
    i = 4
    parsed_range = False # whether t_min and t_max were parsed
    while i < len(sys.argv):
        arg = sys.argv[i]
        if not parsed_range and i + 1 < len(sys.argv): # try to parse t_min and t_max
            try:
                t_min = float(arg)
                t_max = float(sys.argv[i + 1])
                parsed_range = True
                i += 2
                continue
            except Exception: # failed to parse t_min and t_max
                pass
        if arg == "--responsive":
            responsive = True
        elif arg == "--guides":
            guides = True
        elif arg in ("--normalize", "--norm"):
            normalize = True
        elif arg == "--pad" and i + 1 < len(sys.argv): # try to parse pad fraction
            try:
                pad_frac = float(sys.argv[i + 1]); i += 1
            except Exception: # failed to parse pad fraction
                pass
        elif arg == "--save-html" and i + 1 < len(sys.argv):
            save_html_path = sys.argv[i + 1]; i += 1
        elif arg == "--plane":
            show_plane = True
        elif arg == "--arrow" and i + 1 < len(sys.argv): # try to parse arrow scale
            try:
                arrow_scale = float(sys.argv[i + 1]); i += 1
            except Exception: # failed to parse arrow scale
                pass
        elif arg == "--readout-mode" and i + 1 < len(sys.argv):
            readout_mode = sys.argv[i + 1]; i += 1
        elif arg == "--readout-x" and i + 1 < len(sys.argv): # try to parse readout x
            try:
                readout_x = float(sys.argv[i + 1]); i += 1
            except Exception: # failed to parse readout x
                pass
        elif arg == "--readout-y" and i + 1 < len(sys.argv): # see above
            try:
                readout_y = float(sys.argv[i + 1]); i += 1
            except Exception:
                pass
        elif arg == "--readout-size" and i + 1 < len(sys.argv):
            try:
                readout_size = int(sys.argv[i + 1]); i += 1
            except Exception:
                pass
        elif arg == "--readout-xp" and i + 1 < len(sys.argv):
            try:
                readout_xp = float(sys.argv[i + 1]); i += 1
            except Exception:
                pass
        elif arg == "--readout-yp" and i + 1 < len(sys.argv):
            try:
                readout_yp = float(sys.argv[i + 1]); i += 1
            except Exception:
                pass
        elif arg == "--readout-band" and i + 1 < len(sys.argv):
            try:
                readout_band = float(sys.argv[i + 1]); i += 1
            except Exception:
                pass
        elif arg == "--readout-gap" and i + 1 < len(sys.argv):
            try:
                readout_gap = float(sys.argv[i + 1]); i += 1
            except Exception:
                pass
        elif arg == "--readout-where" and i + 1 < len(sys.argv):
            readout_where = sys.argv[i + 1]; i += 1
        i += 1 # next arg

    plot_vector_with_components(funcs, t_min, t_max,
                                responsive=responsive, guides=guides,
                                normalize=normalize, pad_frac=pad_frac,
                                save_html_path=save_html_path, show_plane=show_plane,
                                arrow_scale=arrow_scale,
                                readout_mode=readout_mode,
                                readout_x=readout_x, readout_y=readout_y, readout_size=readout_size,
                                readout_xp=readout_xp, readout_yp=readout_yp, readout_band=readout_band, readout_gap=readout_gap, readout_where=readout_where)