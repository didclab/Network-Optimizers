import React, { Component } from 'react';
import { AppBar, Toolbar, Grid } from "@mui/material";
import Logo from "../assets/images/logo.png";
import { Link } from 'react-router-dom';

class NavbarComponent extends Component {
    render() {
        return (
            <React.Fragment>
                <AppBar className='navbar-root' style={{backgroundColor: "#172753", zIndex: 1400}}>
                    <Toolbar style={{padding: "0 0 0 24px"}}>
                        <Grid container className={"leftNav"} alignItems={"center"}>
                            <Link to={"/metrics"} className={"navbarHome"} >
                                <img className="navbarLogo" src={Logo} alt="One Data Share Logo" />
                                <h4 className="navbarName">One Data Share Metrics Visualization Tool</h4>
                            </Link>
                        </Grid>
                    </Toolbar>
                </AppBar>
            </React.Fragment>
        );
    }
}

export default NavbarComponent; 