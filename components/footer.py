import streamlit as st
from htbuilder import div, p, styles, HtmlElement
from htbuilder.units import percent, px
from htbuilder.func import link

def layout(*args):
    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 80px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1
    )

    body = p(
        id='myFooter',
        style=styles(
            margin=px(0, 0, 0, 0),
            padding=px(5),
            font_size="0.8rem",
            color="rgb(51,51,51)"
        )
    )
    foot = div(
        style=style_div
    )(
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

def footer():
    myargs = [
        "2024 | Made by TeriYumi",
        link("https://github.com/TeriTeriLikeYumi", "@teriyumi"),
    ]
    layout(*myargs)