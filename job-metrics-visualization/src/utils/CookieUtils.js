import Cookies from 'universal-cookie';

const cookies = new Cookies();

export const setCookie = (name, value, options = {}) => {
    cookies.set(name, value, options);
};

export const getCookie = (name) => {
    return cookies.get(name) || null;
};

export const removeCookie = (name, options = {}) => {
    cookies.remove(name, options);
};