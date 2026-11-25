import plotly.graph_objects as go

def make_bullet_comparison(circ_name, circ_rate, nat_ref=23.75):
    gap = circ_rate - nat_ref
    status_color = '#E60000' if gap > 0 else '#16A34A'
    status_text = f"+{gap:.2f} pp" if gap > 0 else f"{gap:.2f} pp"
    verdict = "Higher Churn (Risk)" if gap > 0 else "Lower Churn (Outperforming)"
    
    # Clean Bullet / Compact Comparison Bar
    fig = go.Figure()
    
    # 1. Background range band (National Reference)
    fig.add_trace(go.Bar(
        y=['Benchmark Comparison'],
        x=[nat_ref],
        name='National Reference',
        orientation='h',
        marker=dict(color='rgba(100, 116, 139, 0.25)', line=dict(color='#64748B', width=1)),
        text=[f'National Benchmark: {nat_ref:.2f}%'],
        textposition='inside',
        insidetextanchor='start',
        textfont=dict(size=10.5, color='#334155', family='Inter'),
        hoverinfo='none'
    ))
    
    # 2. Circle Rate Bar (thinner overlay or side-by-side)
    fig.add_trace(go.Bar(
        y=['Benchmark Comparison'],
        x=[circ_rate],
        name=circ_name,
        orientation='h',
        marker=dict(color=status_color, line=dict(color='#FFFFFF', width=1.5)),
        width=0.45,
        text=[f'{circ_name}: {circ_rate:.2f}% ({status_text})'],
        textposition='outside',
        textfont=dict(size=11, color=status_color, family='Inter'),
        cliponaxis=False,
        hovertemplate=f'<b>{circ_name}</b>: %{{x:.2f}}%<br>National: {nat_ref:.2f}%<br>Variance: {status_text}<extra></extra>'
    ))
    
    # 3. Benchmark Target Line (Vertical marker line at 23.75%)
    fig.add_shape(
        type='line',
        x0=nat_ref, x1=nat_ref,
        y0=-0.35, y1=0.35,
        line=dict(color='#1E293B', width=3, dash='solid')
    )
    
    max_range = max(nat_ref, circ_rate) * 1.35
    fig.update_xaxes(range=[0, max_range], ticksuffix='%', gridcolor='#F1F5F9', title_text='Churn Rate (%)')
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=140,
        barmode='overlay',
        margin=dict(l=10, r=20, t=10, b=25),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        showlegend=False
    )
    return fig

b = make_bullet_comparison('Karnataka', 26.10)
print('Bullet comparison created successfully!')
